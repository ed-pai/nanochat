#!/bin/bash
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="/workspace/.cache/nanochat"

source $PWD/.venv/bin/activate

if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# Default number of processes per node
NPROC_PER_NODE=8

# Parse --node argument to set the node rank for distributed training.
# Usage: ./rerun_torch.sh --node=0  or  ./rerun_torch.sh --node 0
NODE_RANK=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --node=*)
            NODE_RANK="${1#*=}"
            shift
            ;;
        --node)
            NODE_RANK="$2"
            shift 2
            ;;
        *)
            # stop parsing on first non-matching arg
            break
            ;;
    esac
done

# Validate NODE_RANK is an integer
if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]]; then
    echo "Invalid --node value: '$NODE_RANK'" >&2
    echo "Usage: $0 [--node N]" >&2
    exit 2
fi

torchrun --nnodes=3 --node_rank=$NODE_RANK --rdzv_id=42 --rdzv_backend=c10d --rdzv_endpoint=192.168.127.6:29400 --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=20 --run=$WANDB_RUN