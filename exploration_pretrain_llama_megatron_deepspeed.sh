#!/bin/bash
set -ex

BASE_PATH=/home/team/xiaohong/llama_demo
DS_CONFIG=${BASE_PATH}/deepspeed_config_llama_megatron_deepspeed.json

TP=4
PP=2
NUM_HOSTS=2
ZERO_STAGE=0
MICRO_BATCH=4

NLAYERS=60
HIDDEN=6656
LEARNING_RATE=0.00015
BATCH_SIZE=1920
HOSTFILE=${BASE_PATH}/deepspeed_hostfile_llama_megatron_deepspeed.txt

MODEL_NAME=llama-30b-c4
LOG_DIR=${BASE_PATH}/log/${MODEL_NAME}-pretrain_ds_z${ZERO_STAGE}_tp${TP}_pp${PP}_nn${NUM_HOSTS}_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}_lr${LEARNING_RATE}_bs${BATCH_SIZE}
OUTPUT_DIR=${BASE_PATH}/models/${MODEL_NAME}/${MODEL_NAME}-pretrain_ds_z${ZERO_STAGE}_tp${TP}_pp${PP}_nn${NUM_HOSTS}_nl${NLAYERS}_hs${HIDDEN}_gb${GLOBAL_BATCH}_mb${MICRO_BATCH}_lr${LEARNING_RATE}_bs${BATCH_SIZE}
INIT_DIR=${BASE_PATH}/models/${MODEL_NAME}/${MODEL_NAME}-pretrain_ds_z${ZERO_STAGE}_tp${TP}_pp${PP}_nn${NUM_HOSTS}_nl${NLAYERS}_hs${HIDDEN}_init
INIT_LOG_DIR=${BASE_PATH}/log/${MODEL_NAME}/${MODEL_NAME}-pretrain_ds_z${ZERO_STAGE}_tp${TP}_pp${PP}_nn${NUM_HOSTS}_nl${NLAYERS}_hs${HIDDEN}_init

mkdir -p $LOG_DIR
mkdir -p $OUTPUT_DIR

cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 1,

  "zero_optimization": {
    "stage": $ZERO_STAGE
  },

  "bf16": {
    "enabled": true
  }
}
EOT


export NCCL_DEBUG=warn
export NCCL_SOCKET_IFNAME=private
export NCCL_IB_DISABLE=1

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"


torchrun --nproc_per_node=8 --nnodes=$NUM_HOSTS --node_rank=1 --master_addr="10.100.7.230" --master_port=6000 exploration_pretrain_llama_megatron_deepspeed.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --ffn-hidden-size 17920 \
    --num-attention-heads 52 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $BATCH_SIZE \
    --train-iters 50000 \
    --optimizer adam \
    --lr $LEARNING_RATE \
    --min-lr 0.000015 \
    --lr-decay-style cosine \
    --log-interval 1 \
    --save-interval 100 \
    --clip-grad 1.0 \
    --lr-warmup-iters 2000 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.02 \
    --layernorm-epsilon 1e-6 \
    --bf16 \
    --checkpoint-activations \
    --tensorboard-dir $LOG_DIR \
    --no-query-key-layer-scaling \
    --dataloader-type single \
    --load $OUTPUT_DIR \
    --save $OUTPUT_DIR \
    $ds_args | tee ${LOG_DIR}/output.log

