import torch
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from datetime import datetime
from time import gmtime, strftime
from functools import partial, lru_cache
import logging
import numpy as np
import glob
import bisect
import os
import random

import deepspeed
from deepspeed.accelerator.real_accelerator import get_accelerator

from megatron import get_args, get_timers
from megatron import get_current_global_batch_size
from megatron import mpu
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron import get_num_microbatches
from megatron import update_num_microbatches
from megatron.optimizer import get_megatron_optimizer
from megatron.initialize import initialize_megatron
from megatron.learning_rates import AnnealingLR
from megatron.model import Float16Module
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model.llama_model import LlamaModel, GPTModelPipe
from megatron.utils import unwrap_model
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import average_losses_across_data_parallel_group
from megatron.schedules import forward_backward_no_pipelining
from megatron.schedules import forward_backward_pipelining_without_interleaving
from megatron.schedules import forward_backward_pipelining_with_interleaving
from megatron.checkpointing import load_checkpoint, get_checkpoint_name

import wandb


def wandb_init():
    args = get_args()
    nnodes = 2
    zero_stage = 0

    # Wait so everyone is done (necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    wandb.init(
        project="llama-30b-c4-pretrain",
        group=f"{nnodes}node-8gpu-zs{zero_stage}-tp{args.tensor_model_parallel_size}-pp{args.pipeline_model_parallel_size}-bs{args.global_batch_size}-mbs{args.micro_batch_size}-{strftime('%Y-%m-%d-%H-%M', gmtime())}",
        job_type="exploration_memory_speed_profile",
        name=f"rank: {torch.distributed.get_rank()}",
        notes=f"{nnodes}node-8gpu-zs{zero_stage}-tp{args.tensor_model_parallel_size}-pp{args.pipeline_model_parallel_size}-bs{args.global_batch_size}-mbs{args.micro_batch_size}",
        config={"task": "pretrain",
                "model": 'llama-30b',
                "batch_size": args.global_batch_size,
                "num-layers": args.num_layers,
                "hidden-size": args.hidden_size,
                "ffn-hidden-size": args.ffn_hidden_size,
                "num-attention-heads": args.num_attention_heads,
                "seq-length": args.seq_length,
                "train-iters": args.train_iters,
                "tensor-model-parallel-size": args.tensor_model_parallel_size,
                "pipeline-model-parallel-size": args.pipeline_model_parallel_size,
                "zero-stage": zero_stage,
                "nnodes": nnodes,
                "nproc_per_node": 8}
    )


# from megatron.checkpointing.print_rank_0
def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


# from megatron.checkpointing.is_rank_0
def is_rank_0():
    """Check whether it is rank 0. For AML, check if it is rank 0 of a node"""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            return True
        else:
            return False
    else:
        return True


# modify from megatron.checkpointing.save_checkpoint, to be able to save for release
def ensure_directory_exists(filename):
    """Build filename's path if it does not already exists."""
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_checkpoint_tracker_filename(checkpoints_path):
    """Tracker file rescords the latest chckpoint during
    training to restart from."""
    return os.path.join(checkpoints_path, 'latest_checkpointed_iteration.txt')


def save_checkpoint(iteration, model, optimizer, lr_scheduler, release=False):
    """Save a model checkpoint."""
    args = get_args()

    # Only rank zero of the data parallel writes to the disk.
    if not args.deepspeed:
        model = unwrap_model(model)

    print_rank_0('saving checkpoint at iteration {:7d} to {}'.format(
        iteration, args.save))

    if not torch.distributed.is_initialized() or mpu.get_data_parallel_rank() == 0 \
            or args.deepspeed:

        # Arguments, iteration, and model.
        state_dict = {}
        state_dict['args'] = args
        state_dict['checkpoint_version'] = 3.0
        state_dict['iteration'] = iteration
        state_dict['tokens'] = args.consumed_train_tokens

        # DeepSpeed saves the model/optimizer/scheduler
        if not args.deepspeed:
            if len(model) == 1:
                state_dict['model'] = model[0].state_dict_for_save_checkpoint()
            else:
                for i in range(len(model)):
                    mpu.set_virtual_pipeline_model_parallel_rank(i)
                    state_dict['model%d' % i] = model[i].state_dict_for_save_checkpoint()

            # Optimizer stuff.
            if not args.no_save_optim:
                if optimizer is not None:
                    state_dict['optimizer'] = optimizer.state_dict()
                if lr_scheduler is not None:
                    state_dict['lr_scheduler'] = lr_scheduler.state_dict()

        # RNG states.
        if not args.no_save_rng:
            state_dict['random_rng_state'] = random.getstate()
            state_dict['np_rng_state'] = np.random.get_state()
            state_dict['torch_rng_state'] = torch.get_rng_state()
            state_dict['cuda_rng_state'] = get_accelerator().get_rng_state()
            state_dict['rng_tracker_states'] \
                = mpu.get_cuda_rng_tracker().get_states()

        # Save.
        checkpoint_name = get_checkpoint_name(args.save, iteration, release)
        if not args.deepspeed:
            ensure_directory_exists(checkpoint_name)
            torch.save(state_dict, checkpoint_name)

    if args.deepspeed:
        # megatron model uses state_dict_for_save_checkpointing instead of the standard state_dict
        # state_dict is used by deepspeed for module saving so it needs to point to the right function
        if args.no_pipeline_parallel:
            original_state_dict = model[0].module.state_dict
            model[0].module.state_dict = model[0].module.state_dict_for_save_checkpoint

        # Saving is a collective communication
        checkpoint_name = get_checkpoint_name(args.save, iteration, release)

        # Trim off the filename and mp_rank_* directory.
        for _ in range(3):
            checkpoint_name = os.path.dirname(checkpoint_name)
        model[0].save_checkpoint(checkpoint_name, client_state=state_dict)

        if args.no_pipeline_parallel:
            model[0].module.state_dict = original_state_dict

    # Wait so everyone is done (necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    print_rank_0('  successfully saved checkpoint at iteration {:7d} to {}'.format(
        iteration, args.save))

    # And update the latest iteration
    if is_rank_0():
        tracker_filename = get_checkpoint_tracker_filename(args.save)
        with open(tracker_filename, 'w') as f:
            if release:
                f.write('release')
            else:
                f.write(str(iteration))

    # Wait so everyone is done (not necessary)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


# from megatron.training.get_model
def get_model(model_provider_func):
    """Build the model."""
    args = get_args()

    # Build model.
    if mpu.get_pipeline_model_parallel_world_size() > 1 and \
            args.virtual_pipeline_model_parallel_size is not None:
        model = []
        for i in range(args.virtual_pipeline_model_parallel_size):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            this_model = model_provider_func(
                pre_process=pre_process,
                post_process=post_process
            )
            model.append(this_model)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        model = model_provider_func(
            pre_process=pre_process,
            post_process=post_process
        )

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            mpu.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on (tensor, pipeline) '
              'model parallel rank ({}, {}): {}'.format(
            mpu.get_tensor_model_parallel_rank(),
            mpu.get_pipeline_model_parallel_rank(),
            sum([sum([p.ds_numel if hasattr(p, 'ds_id') else p.nelement() for p in model_module.parameters()])
                 for model_module in model])), flush=True)

    if args.deepspeed:
        return model

    # GPU allocation.
    for model_module in model:
        model_module.to(get_accelerator().current_device_name())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        model = [Float16Module(model_module, args) for model_module in model]

    if args.DDP_impl == 'torch':
        i = get_accelerator().current_device()
        model = [torchDDP(model_module, device_ids=[i], output_device=i,
                          process_group=mpu.get_data_parallel_group())
                 for model_module in model]
        return model

    if args.DDP_impl == 'local':
        model = [LocalDDP(model_module,
                          args.accumulate_allreduce_grads_in_fp32,
                          args.use_contiguous_buffers_in_ddp)
                 for model_module in model]
        return model

    raise NotImplementedError('Unknown DDP implementation specified: {}. '
                              'Exiting.'.format(args.DDP_impl))


# from megatron.training.get_learning_rate_scheduler
def update_train_iters(args):
    # For iteration-based training, we don't need to do anything
    if args.train_iters:
        return

    # Constant batch size with sample-based training.
    if args.rampup_batch_size is None:
        args.train_iters = args.train_samples // args.global_batch_size

    else:
        # Sample based training with rampup batch size.
        iterations = 0
        consumed_samples = 0
        # Rampup phase.
        while consumed_samples <= int(args.rampup_batch_size[2]):
            update_num_microbatches(consumed_samples, consistency_check=False)
            consumed_samples += get_current_global_batch_size()
            iterations += 1
        # Reset
        update_num_microbatches(0, consistency_check=False)
        # Constant phase
        # Note that we throw away any partial last batch.
        iterations += (args.train_samples - consumed_samples) // \
                      args.global_batch_size
        args.train_iters = iterations

    print_rank_0('setting training iterations to {}'.format(args.train_iters))


def get_learning_rate_scheduler(optimizer):
    """Build the learning rate scheduler."""
    args = get_args()

    # Iteration-based training.
    if args.train_iters:
        if args.lr_decay_iters is None:
            args.lr_decay_iters = args.train_iters
        decay_steps = args.lr_decay_iters * args.global_batch_size
        if args.lr_warmup_fraction is not None:
            warmup_steps = args.lr_warmup_fraction * decay_steps
        else:
            warmup_steps = args.lr_warmup_iters * args.global_batch_size
    # Sample-based training.
    elif args.train_samples:
        # We need to set training iters for later use. Technically
        # we need to adjust the training samples too (due to last
        # batch being incomplete) but we leave it as is for now.
        update_train_iters(args)
        if args.lr_decay_samples is None:
            args.lr_decay_samples = args.train_samples
        decay_steps = args.lr_decay_samples
        if args.lr_warmup_fraction is not None:
            warmup_steps = args.lr_warmup_fraction * decay_steps
        else:
            warmup_steps = args.lr_warmup_samples
    else:
        raise Exception(
            'either train-iters or train-samples should be provided.')

    lr_scheduler = AnnealingLR(
        optimizer,
        max_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        decay_style=args.lr_decay_style,
        use_checkpoint_lr_scheduler=args.use_checkpoint_lr_scheduler,
        override_lr_scheduler=args.override_lr_scheduler)

    return lr_scheduler


# borrow from https://github.com/facebookresearch/fairseq/blob/main/fairseq/data/indexed_dataset.py
def _warmup_mmap_file(path):
    with open(path, "rb") as stream:
        while stream.read(100 * 1024 * 1024):
            pass


class MMapIndexedDataset(torch.utils.data.Dataset):
    def __init__(self, path, pretrain_max_seq_len=2048, _bin_buffer_size_multiplier=4, dtype="int32"):
        logging.warning(f"{datetime.now().strftime('%H:%M:%S')} Loading data from {path}...")
        super().__init__()
        self._path = path
        self.pretrain_max_seq_len = pretrain_max_seq_len
        self._bin_buffer_size_multiplier = _bin_buffer_size_multiplier
        self.dtype = dtype
        _warmup_mmap_file(path)
        self._bin_buffer_mmap = np.memmap(self._path, mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)
        logging.warning(f"{datetime.now().strftime('%H:%M:%S')} loaded total {self.__len__()} samples...")

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap

    def __len__(self):
        return int(len(self._bin_buffer_mmap) / self.pretrain_max_seq_len / self._bin_buffer_size_multiplier)

    @lru_cache(maxsize=8)
    def __getitem__(self, ind):
        np_array = np.frombuffer(self._bin_buffer
                                 , dtype=self.dtype
                                 , count=self.pretrain_max_seq_len
                                 , offset=ind * self.pretrain_max_seq_len * self._bin_buffer_size_multiplier)
        return torch.tensor(np_array, dtype=torch.long)


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, "datasets should not be an empty iterable"
        self.datasets = list(datasets)
        self.real_sizes = [len(d) for d in self.datasets]
        self.cumulative_sizes = np.cumsum(self.real_sizes)
        logging.warning(
            f"{datetime.now().strftime('%H:%M:%S')} loaded total {len(self.cumulative_sizes)} datasets, {self.cumulative_sizes[-1]} samples...")

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx, sample_idx = self._get_dataset_and_sample_index(idx)
        input_ids = self.datasets[dataset_idx][sample_idx]
        return dict(input_ids=input_ids)

    def _get_dataset_and_sample_index(self, idx: int):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % self.real_sizes[dataset_idx]  # no need this line??
        return dataset_idx, sample_idx


def model_provider_func(pre_process=True, post_process=True):
    """Build the model."""

    args = get_args()
    with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
        if args.deepspeed and not args.no_pipeline_parallel:
            model = GPTModelPipe()
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe

            # Predompute the attention mask and store it in args. This avoids having to
            # pipeline it as an activation during training. The mask is constant, and thus
            # we can reuse it.
            attention_mask = torch.tril(torch.ones((1, args.seq_length - 1, args.seq_length - 1),
                                                   device=get_accelerator().current_device_name())).view(1, 1,
                                                                                                         args.seq_length - 1,
                                                                                                         args.seq_length - 1)

            # Convert attention mask to binary:
            attention_mask = (attention_mask < 0.5)
            if args.fp16:
                attention_mask = attention_mask.half()
            elif args.bf16:
                attention_mask = attention_mask.bfloat16()

            # Attention mask must be bool.
            args.attn_mask = attention_mask.to(torch.bool)

        else:
            model = LlamaModel(
                parallel_output=True,
                add_pooler=False,
                pre_process=pre_process,
                post_process=post_process
            )
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    eod_token = 2  # TODO remove this hard coding

    # Items and their type.
    keys = ['input_ids']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['input_ids'].long()
    labels = tokens_[:, 1:args.seq_length].contiguous()
    tokens = tokens_[:, :args.seq_length - 1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, _ = get_ltor_masks_and_position_ids(
        tokens,
        eod_token,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask


def get_batch_pipe(data):
    args = get_args()
    eod_token = 2  # TODO remove this hard coding
    # tokenizer = get_tokenizer()
    # eod_token = tokenizer.eod

    keys = ['input_ids']
    datatype = torch.int64
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['input_ids'].long()
    labels = tokens_[:, 1:args.seq_length].contiguous()
    tokens = tokens_[:, :args.seq_length - 1].contiguous()

    # Get the masks
    # TODO, for pipeline parallel, for pretraining, anyway to remove network of attention_mask/loss_mask/labels??
    attention_mask, loss_mask, _ = get_ltor_masks_and_position_ids(
        tokens,
        eod_token,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return (tokens, attention_mask), (labels, loss_mask)


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat(
        [loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses, group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / torch.distributed.get_world_size(group=mpu.get_data_parallel_group())

    return averaged_losses


def loss_func(loss_mask, output_tensor):
    args = get_args()
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    return loss, {'lm loss': averaged_loss[0]}


def forward_step_func(data_iterator, model):
    """Forward step."""
    args = get_args()
    # Get the batch.
    timers = get_timers()
    timers('batch-generator').start()
    tokens, labels, loss_mask, attention_mask = get_batch(data_iterator)
    timers('batch-generator').stop()
    output_tensor = model(tokens, attention_mask, labels=labels)
    # Output_tensor stores the standard loss, loos_func calculates the total loss.
    return output_tensor, partial(loss_func, loss_mask)


def pretrain():
    ### Note ###
    # 1. deepspeed zero-1 doesnt work with bf16: https://github.com/microsoft/DeepSpeed/issues/1835, have to use zero-0 (zero disabled?) as a hack
    args = get_args()

    ### get model, optimizer, lr_scheduler ###
    model = get_model(model_provider_func)
    unwrapped_model = unwrap_model(model, (torchDDP, LocalDDP, Float16Module))
    optimizer = get_megatron_optimizer(unwrapped_model)
    lr_scheduler = get_learning_rate_scheduler(optimizer)

    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")
        model, optimizer, _, lr_scheduler = deepspeed.initialize(model=model[0], optimizer=optimizer, args=args,
                                                                 lr_scheduler=lr_scheduler,
                                                                 mpu=mpu if args.no_pipeline_parallel else None)
        if not args.no_pipeline_parallel:
            # hack to get batch_fn from pretrain_gpt.py
            model.set_batch_fn(model.module._megatron_batch_fn)

            assert model.grid.get_pipe_parallel_rank() == mpu.get_pipeline_model_parallel_rank()
            assert model.grid.get_slice_parallel_rank() == mpu.get_tensor_model_parallel_rank()
            assert model.grid.get_data_parallel_rank() == mpu.get_data_parallel_rank()
        model = [model]

    if args.load:
        # To start pretraining fresh from init checkpoint, with args.deepseepd (ds_pp): set args.finetune (so that lr_scheduler and optimizer will not be loaded)
        # To resume pretraining, with args.deepseepd: no extra setup, (load from resuming checkpoint, will automatically load model weights, optimizer, learning rate scheduler, random state, consumed_train_tokens, iterations)
        # To start pretraining fresh from init checkpoint, without args.deepseepd (no ds_pp): no extra setup (load from release folder, will automatically load only model weights)
        # To resume pretraining, without args.deepseepd: no extra setup, (load from resuming checkpoint, will automatically load model weights, optimizer, learning rate scheduler, random state, consumed_train_tokens, iterations)
        # TODO model is saved on different servers and needs to be copied over to all servers for recover; how to optimize?
        timers = get_timers()
        torch.distributed.barrier()
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler, strict=True)
        torch.distributed.barrier()
    elif args.save:
        # To save init checkpoint
        args.iteration = 0
        save_checkpoint(args.iteration, model, optimizer, lr_scheduler, release=True)
        return
    else:
        args.iteration = 0
        timers = get_timers()

    # We only support local DDP with multiple micro-batches. TODO local ddp vs torch ddp? torch ddp doesnt support accumulate_allreduce_grads_in_fp32, anything else?
    if len(model) > 1 or mpu.get_pipeline_model_parallel_world_size() > 1:
        assert args.DDP_impl == 'local'

    ### get dataset ###
    data_files = sorted(glob.glob("/home/team/xiaohong/llama_demo/data/tokenized_corpus_c4*"))[1:]
    train_dataset = ConcatDataset([MMapIndexedDataset(file) for file in data_files])
    # TODO data_files should be pre shuffled, and data loader should load one by one in sequence, i.e. data should be determined by global_steps;
    # TODO continue data loading from desired iteration
    # TODO Streaming data? so recover from failure is fast in loading data
    train_dataloader = build_pretraining_data_loader(train_dataset, consumed_samples=0)
    train_data_iterator = iter(train_dataloader)

    ### Start Training ###
    for model_module in model:
        model_module.train()

    iteration = args.iteration
    while (iteration < args.train_iters) and (
            args.train_tokens is None or args.consumed_train_tokens < args.train_tokens):
        update_num_microbatches(args.consumed_train_samples)
        if args.deepspeed:
            global_batch_size = mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
            model[0].set_train_batch_size(global_batch_size)
        if args.deepspeed and (not args.no_pipeline_parallel):
            loss = model[0].train_batch(data_iter=train_data_iterator)  # uses deepspeed pipeline parallel
            print(f"{datetime.now().strftime('%H:%M:%S')} iteration: {iteration}, losses_reduced: {loss}")
            iteration = iteration + 1
            wandb.log({"loss": loss, "learning_rate": lr_scheduler.get_lr()})
        else:
            # Set grad to zero.
            if not args.deepspeed:
                if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_ddp:
                    for partition in model:
                        partition.zero_grad_buffer()
                else:
                    optimizer.zero_grad()

            if mpu.get_pipeline_model_parallel_world_size() > 1:
                if args.virtual_pipeline_model_parallel_size is not None:
                    forward_backward_func = forward_backward_pipelining_with_interleaving
                    assert get_num_microbatches() % args.pipeline_model_parallel_size == 0, \
                        'number of microbatches is not divisible by pipeline-parallel ' \
                        'size when using interleaved schedule'
                else:
                    forward_backward_func = forward_backward_pipelining_without_interleaving
            else:
                forward_backward_func = forward_backward_no_pipelining

            losses_reduced = forward_backward_func(forward_step_func, train_data_iterator, model, optimizer, timers,
                                                   forward_only=False)

            # All-reduce if needed.
            if not args.deepspeed and args.DDP_impl == 'local':
                timers('backward-params-all-reduce').start()
                for model_module in model:
                    model_module.allreduce_gradients()
                timers('backward-params-all-reduce').stop()

            # All-reduce word_embeddings' grad across first and last stages to ensure
            # that word_embeddings parameters stay in sync.
            # This should only run for models that support pipelined model parallelism
            # (normally if pipeline parallel, can use deepspeed's pipeline parallel, can skip below)
            timers('backward-embedding-all-reduce').start()
            if not args.deepspeed:
                if (mpu.is_pipeline_first_stage(ignore_virtual=True) or
                    mpu.is_pipeline_last_stage(ignore_virtual=True)) and \
                        mpu.get_pipeline_model_parallel_world_size() > 1:
                    if mpu.is_pipeline_first_stage(ignore_virtual=True):
                        unwrapped_model = model[0]
                    elif mpu.is_pipeline_last_stage(ignore_virtual=True):
                        unwrapped_model = model[-1]
                    unwrapped_model = unwrap_model(
                        unwrapped_model, (torchDDP, LocalDDP, Float16Module))

                    if unwrapped_model.share_word_embeddings:
                        word_embeddings_weight = unwrapped_model.word_embeddings_weight()
                        if args.DDP_impl == 'local':
                            grad = word_embeddings_weight.main_grad
                        else:
                            grad = word_embeddings_weight.grad
                        torch.distributed.all_reduce(grad, group=mpu.get_embedding_group())
            timers('backward-embedding-all-reduce').stop()

            # Update parameters.
            timers('optimizer').start()
            if args.deepspeed:
                increment = get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
                model[0].step(lr_kwargs={'increment': increment})
                timers('optimizer').stop()
                print(
                    f"{datetime.now().strftime('%H:%M:%S')} iteration: {iteration}, losses_reduced: {losses_reduced[0]}")
                iteration = iteration + 1
                wandb.log({"loss": losses_reduced[0], "learning_rate": lr_scheduler.get_lr()})
            else:
                update_successful, _, _ = optimizer.step()
                timers('optimizer').stop()
                if update_successful:
                    increment = get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
                    lr_scheduler.step(increment=increment)
                    print(
                        f"{datetime.now().strftime('%H:%M:%S')} iteration: {iteration}, losses_reduced: {losses_reduced[0]}")
                    iteration = iteration + 1
                    wandb.log({"loss": losses_reduced[0], "learning_rate": lr_scheduler.get_lr()})
                else:
                    print(
                        f"{datetime.now().strftime('%H:%M:%S')} iteration: {iteration}, updated failed, iteration skipped")

        args.iteration = iteration
        new_samples = mpu.get_data_parallel_world_size() * args.micro_batch_size * get_num_microbatches()
        args.consumed_train_samples += new_samples
        args.consumed_train_tokens += new_samples * args.seq_length

        if ((
            iteration) % args.save_interval == 0) and args.save:  # TODO why pp=2, mp=2, dp=2, have 8 shards of optim_states?
            save_checkpoint(iteration, model, optimizer, lr_scheduler)

    if ((iteration) % args.save_interval != 0) and args.save:
        save_checkpoint(iteration, model, optimizer, lr_scheduler)


def init():
    ### Init ###
    initialize_megatron(args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
    args = get_args()
    print(f"{datetime.now().strftime('%H:%M:%S')} args: {args}")
    wandb_init()


# TODO
# megatron's virtual_pipeline_model_parallel_size. why need virtual pipeline parallel? only when memory not enough to load all tensor and pipeline paralleled layers into GPU, and cpu offloading is enabled? uses special forward_backward_pipelining_with_interleaving? Handling not included in this code. there is no virtual parallel concept for deepspeed's pipeline parallel?
# other curriculum_learning, compression_training?
# Analyze training speed/memory performance: 1. all same, increasing batchsize, not increasing speed? 2. checkpoint-activations, distribute-checkpointed-activations not increasing speed?
# https://github.com/microsoft/Megatron-DeepSpeed/issues/105, as such, many new improvements from megatron including sequence parallel, fast attention is not supported yet here...  probably shift to Megatron?
# ALiBi positional embedding and not supported by megatron-deepspeed (or megatron) yet
# Lion optimizer not supported by megatron-deepspeed (or megatron) yet. Only SGD and ADAM (using apex's Fused implementation)

if __name__ == "__main__":
    init()
    pretrain()
    wandb.finish()
