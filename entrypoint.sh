#!/bin/bash
# Node rank: 0 for master, ... for slaves
# --rdvz-id=$JOB_ID \

echo "START TIME: $(date)"

export NCCL_IB_GID_INDEX=$(grep 'RoCE v2' $(grep '0000:0000:0000:0000:0000:ffff' /sys/class/infiniband/mlx5_bond_0/ports/1/gids/* | cut -d ':' -f 1 | sed 's/gids/gid_attrs\/types/') |  sed -e 's/.*\/\([0-9]*\):.*/\1/')
export NCCL_IB_HCA=mlx5_bond_
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=$RUNAI_NUM_OF_GPUS

export NCCL_DEBUG=INFO
export PROCESSES_PER_NODE=$RUNAI_NUM_OF_GPUS

export ACCELERATE_DEEPSPEED_ZERO3_INIT=true

pip install "huggingface_hub[hf_transfer]"
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
pip install -r requirements.txt
 
# Display the role of the current node
echo "Role: $(hostname -s | tr -dc '0-9')"
echo "Num workers: $WORLD_SIZE"
echo "rdzv endpoint: $MASTER_ADDR:$MASTER_PORT"

# Run the command
torchrun \
	--nnodes=$WORLD_SIZE \
	--rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
	--rdzv_backend=c10d \
	--nproc-per-node=$PROCESSES_PER_NODE \
	--role $(hostname -s|tr -dc '0-9'): \
	--max-restarts=0 \
	--tee 3 \
	main.py --config config/config_rcp.yaml >> training.out 2>&1
EXIT_CODE=$?

# Cleanup duty (the job may have failed or succeeded)
echo "END TIME: $(date)"

# Finally exit with the same error code as the torchrun command
exit $EXIT_CODE

