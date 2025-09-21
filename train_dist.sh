%%writefile train_dist.sh
#!/bin/bash
GPU_NUM=$1
CFG=$2
DATASETS=$3
OUTPUT_DIR=$4
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
# SỬA LỖI: Tự động chọn một cổng mạng ngẫu nhiên để tránh lỗi "address already in use"
PORT=$(shuf -i 10000-65500 -n 1)
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
PRETRAIN_MODEL_PATH=${PRETRAIN_MODEL_PATH:-"weights/groundingdino_swint_ogc.pth"}
TEXT_ENCODER_TYPE=${TEXT_ENCODER_TYPE:-"bert-base-uncased"}

echo "
GPU_NUM = $GPU_NUM
CFG = $CFG
DATASETS = $DATASETS
OUTPUT_DIR = $OUTPUT_DIR
NNODES = $NNODES
NODE_RANK = $NODE_RANK
PORT = $PORT
MASTER_ADDR = $MASTER_ADDR
PRETRAIN_MODEL_PATH = $PRETRAIN_MODEL_PATH
TEXT_ENCODER_TYPE = $TEXT_ENCODER_TYPE
"

# SỬA LỖI: Sử dụng torchrun thay vì torch.distributed.launch đã lỗi thời
# Thêm --standalone và --nnodes=1 để đảm bảo nó chạy ổn định trên một máy duy nhất (như Colab)
torchrun \
    --nproc_per_node="${GPU_NUM}" \
    --master_port="${PORT}" \
    --standalone \
    --nnodes=1 \
    main.py \
    --output_dir "${OUTPUT_DIR}" \
    -c "${CFG}" \
    --datasets "${DATASETS}"  \
    --pretrain_model_path "${PRETRAIN_MODEL_PATH}" \
    --options text_encoder_type="$TEXT_ENCODER_TYPE"
