# Borrow from https://github.com/mosaicml/examples/blob/3e47a86747f531719d1e581487c73049b66a53b4/examples/common/optim
# TODO FusedLionW (https://github.com/NVIDIA/apex/blob/master/apex/optimizers/fused_adam.py, https://github.com/NVIDIA/apex/blob/8b7a1ff183741dd8f9b87e7bafd04cfde99cea28/csrc/multi_tensor_adam.cu)
import math
from typing import Callable, Optional, Tuple
import torch
from torch.optim.optimizer import Optimizer
from megatron import mpu
import torch.distributed as dist
import collections

class DecoupledLionW(Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-4,
            betas: Tuple[float, float] = (0.9, 0.99),
            weight_decay: float = 0.0,
    ):
        if lr <= 0.:
            raise Exception(f"Invalid LR: {lr}. LR must be > 0")
        if not all([0. <= beta <= 1. for beta in betas]):
            raise Exception(f"Invalid beta values: {betas} All betas must be between 0 and 1.")
        if weight_decay >= 1e-3:
            print(
                f'You are using a high value of `weight_decay={weight_decay}` for the `DecoupledLionW` optimizer. Are you sure you want to do this? '
                f'Your model\'s weights will be multiplied by {1.0 - weight_decay} on every step!'
            )

        defaults = {'lr': lr, 'betas': betas, 'weight_decay': weight_decay}

        super().__init__(params, defaults)

        for group in self.param_groups:
            group['initial_lr'] = group['lr']

    @staticmethod
    def lionw(p, grad, exp_avg, lr, initial_lr, wd, beta1, beta2) -> None:
        # stepweight decay
        if wd != 0:
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            p.data.mul_(1 - decay_factor * wd)

        # update is interpolation between gradient and momentum
        update = exp_avg.lerp(grad, 1 - beta1).sign_()
        p.add_(update, alpha=-lr)

        # momentum is interp b/w gradient and itself
        exp_avg.lerp_(grad, 1 - beta2)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: p.grad is not None and p.requires_grad,
                            group['params']):

                grad, lr, initial_lr, wd, beta1, beta2, state = p.grad, group['lr'], group['initial_lr'], group['weight_decay'], *group['betas'], self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']

                self.lionw(p, grad, exp_avg, lr, initial_lr, wd, beta1, beta2)

        return loss


class OutlierDetector:
    def __init__(self,
                 threshold: float = 7.5,
                 delay_interval: int = 500):

        self.intermediate_data_queue = collections.deque(maxlen=delay_interval)
        self.delayed_moving_average = collections.deque(maxlen=delay_interval)
        self.threshold = threshold

    def insert_observation(self, obs: float) -> bool:
        if len(self.intermediate_data_queue) >= self.intermediate_data_queue.maxlen:
            # move data from intermediate queue to slow moving average queue
            intermediate_obs = self.intermediate_data_queue.popleft()
            self.delayed_moving_average.append(intermediate_obs)

        self.intermediate_data_queue.append(obs)
        delayed_mva = self.get_delayed_mva()
        return delayed_mva is not None and obs > self.threshold * delayed_mva

    def get_delayed_mva(self):
        if len(self.delayed_moving_average) > 0:
            return sum(self.delayed_moving_average) / len(self.delayed_moving_average)
        else:
            return None

class DecoupledAdaLRLion(Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-4,
            betas: Tuple[float, float] = (0.9, 0.99),
            weight_decay: float = 0.0,
            outlier_threshold: float = 10.0,
            timeout: int = 100,
            lr_penalty: float = .707,
            min_scale: float = 1e-4
    ):
        if lr <= 0.:
            raise Exception(f"Invalid LR: {lr}. LR must be > 0")
        if not all([0. <= beta <= 1. for beta in betas]):
            raise Exception(f"Invalid beta values: {betas} All betas must be between 0 and 1.")
        if weight_decay >= 1e-3:
            print(
                f'You are using a high value of `weight_decay={weight_decay}` for the `DecoupledLionW` optimizer. Are you sure you want to do this? '
                f'Your model\'s weights will be multiplied by {1.0 - weight_decay} on every step!')

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )

        super().__init__(params, defaults)

        for group in self.param_groups:
            group['initial_lr'] = group['lr']
        self.outlier_threshold = outlier_threshold
        self.timeout = timeout
        self.lr_penalty = lr_penalty
        self.min_scale = min_scale

    @staticmethod
    def lionw(p, grad, exp_avg, lr, initial_lr, wd, beta1, beta2) -> None:
        # stepweight decay
        if wd != 0:
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            p.data.mul_(1 - decay_factor * wd)

        # update is interpolation between gradient and momentum
        update = exp_avg.lerp(grad, 1 - beta1).sign_()
        p.add_(update, alpha=-lr)

        # momentum is interp b/w gradient and itself
        exp_avg.lerp_(grad, 1 - beta2)

    @staticmethod
    def adjust_lr(lr: float, lr_penalty: float, num_times: int, min_scale: float):
        return lr * max(min_scale, lr_penalty ** num_times)

    @torch.no_grad()
    def step(
            self,
            closure: Optional[Callable] = None
    ):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: p.grad is not None and p.requires_grad, group['params']):

                grad, lr, initial_lr, wd, beta1, beta2, state = p.grad, group['lr'], group['initial_lr'], group['weight_decay'], *group['betas'], self.state[p]

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['moment_tracker'] = OutlierDetector(self.outlier_threshold)
                    state['outlier_timestamp'] = []
                    state['step'] = 0

                exp_avg = state['exp_avg']

                # determine if the new moment resulting from this grad would be an outlier
                moment_norm = torch.linalg.vector_norm(
                    exp_avg.lerp(grad, 1 - beta2)
                ) ** 2

                if dist.get_world_size() > 1:
                    dist.all_reduce(moment_norm, op=dist.ReduceOp.SUM)
                moment_norm = math.sqrt(moment_norm)

                if state['moment_tracker'].insert_observation(moment_norm):
                    state['outlier_timestamp'].append(state['step'])

                removed = []
                for ts in state['outlier_timestamp']:
                    if state['step'] - ts > self.timeout:
                        removed.append(ts)

                for ts in removed:
                    state['outlier_timestamp'].remove(ts)

                lr = self.adjust_lr(lr, self.lr_penalty, len(state['outlier_timestamp']), self.min_scale)
                self.lionw(
                    p,
                    grad,
                    exp_avg,
                    lr,
                    initial_lr,
                    wd,
                    beta1,
                    beta2
                )
                state['step'] += 1

        return loss

class DecoupledClipLion(Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-4,
            betas: Tuple[float, float] = (0.9, 0.99),
            weight_decay: float = 0.0,
            outlier_threshold=5.0
    ):
        if lr <= 0.:
            raise Exception(f"Invalid LR: {lr}. LR must be > 0")
        if not all([0. <= beta <= 1. for beta in betas]):
            raise Exception(f"Invalid beta values: {betas} All betas must be between 0 and 1.")
        if weight_decay >= 1e-3:
            print(
                f'You are using a high value of `weight_decay={weight_decay}` for the `DecoupledLionW` optimizer. Are you sure you want to do this? '
                f'Your model\'s weights will be multiplied by {1.0 - weight_decay} on every step!')

        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay
        )

        super().__init__(params, defaults)

        for group in self.param_groups:
            group['initial_lr'] = group['lr']
        self.outlier_threshold = outlier_threshold

    @staticmethod
    def lionw(p, grad, exp_avg, lr, initial_lr, wd, beta1, beta2) -> None:
        # stepweight decay
        if wd != 0:
            decay_factor = (lr / initial_lr) if initial_lr else 1.0
            p.data.mul_(1 - decay_factor * wd)

        # update is interpolation between gradient and momentum
        update = exp_avg.lerp(grad, 1 - beta1).sign_()
        p.add_(update, alpha=-lr)

        # momentum is interp b/w gradient and itself
        exp_avg.lerp_(grad, 1 - beta2)

    @torch.no_grad()
    def step(
            self,
            closure: Optional[Callable] = None
    ):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: p.grad is not None and p.requires_grad, group['params']):

                grad, lr, initial_lr, wd, beta1, beta2, state = p.grad, group['lr'], group['initial_lr'], group['weight_decay'], *group['betas'], self.state[p]

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['grad_tracker'] = OutlierDetector(self.outlier_threshold)
                    state['clipped_batches'] = torch.tensor(0.0)

                exp_avg = state['exp_avg']

                # determine if the new moment resulting from this grad would be an outlier
                grad_norm = torch.linalg.vector_norm(
                    grad
                ) ** 2

                if dist.get_world_size() > 1:
                    dist.all_reduce(moment_norm, op=dist.ReduceOp.SUM)
                grad_norm = math.sqrt(grad_norm)

                if state['grad_tracker'].insert_observation(grad_norm):
                    state['clipped_batches'] += 1.0
                    clip_norm = state['grad_tracker'].get_slow_mva() * self.outlier_threshold
                    grad = grad.div(grad_norm).mul_(clip_norm)

                self.lionw(
                    p,
                    grad,
                    exp_avg,
                    lr,
                    initial_lr,
                    wd,
                    beta1,
                    beta2
                )

        return loss