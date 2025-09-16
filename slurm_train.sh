#!/bin/bash
#SBATCH --job-name multimeditron

#SBATCH --chdir /users/$USERNAME/meditron/axolotl
#SBATCH --output /users/$USERNAME/meditron/reports/R-%x.%j.out
#SBATCH --error /users/$USERNAME/meditron/reports/R-%x.%j.err
#SBATCH --nodes 4                   # number of Nodes
#SBATCH --ntasks-per-node 1         # number of MP tasks. IMPORTANT: torchrun represents just 1 Slurm task
#SBATCH --gpus-per-node=4           # Number of GPUs
#SBATCH --cpus-per-task 288         # number of CPUs per task.
#SBATCH --time 11:59:59             # maximum execution time (DD-HH:MM:SS)
#SBATCH -A a06

mkdir -p logs

if [[ " $@ " =~ " --pack "  && "${SLURM_NTASKS}" -gt 1 ]]; then
    echo "Packing datasets is not yet supported with a multi-rank setup (requires splitting datasets across ranks)"
    echo "Use with sbatch -N 1 -n 1 --ntasks-per-node 1 --cpus-per-task 288 slurm_train.sh --pack"
    echo "Limit the number of processed datasets through the config file (processing is in parallel)"
    exit 1
fi

export WANDB_DIR=/store/swissai/a06/meditron/wandb
export WANDB_ENTITY=${USER}
export WANDB_PROJECT=${SLURM_JOB_NAME}
export WANDB_API_KEY=$(cat ${CAPSCRATCH:-$SCRATCH}/.wandb_api_key)
export WANDB_MODE="offline"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_LAUNCH_BLOCKING=1

echo "START TIME: $(date)"
# auto-fail on any errors in this script
set -eo pipefail
# logging script's variables/commands for future debug needs
set -x
######################
### Set enviroment ###
######################
echo "NODES: $SLURM_NNODES"
######## Args ########
export HF_HOME=/store/swissai/a06/meditron/hf
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
######################
######################
#### Set network #####
######################
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
######################
# note that we don't want to interpolate `\$SLURM_PROCID` till `srun` since otherwise all nodes will get
# 0 and the launcher will hang
#
# same goes for `\$(hostname -s|tr -dc '0-9')` - we want it to interpolate at `srun` time

# pip3 install -e /users/$USERNAME/meditron/axolotl/DeepSpeed-Kernels && \

LAUNCHER="
    torchrun \
    --nproc_per_node $SLURM_GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank \$SLURM_PROCID \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "

# export CMD="$LAUNCHER main.py"
# echo $CMD
# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# SRUN_ARGS=" \
#     --wait 60 \
#     "

export ACCELERATE_DEEPSPEED_ZERO3_INIT=false
export HF_HUB_ENABLE_HF_TRANSFER=1
export ENROOT_ENTRYPOINT=$(which enroot-entrypoint.sh 2> /dev/null || echo env/enroot-entrypoint.sh)
export BENCHY_CONFIG_FILE=config/benchy.yaml
export BENCHY_OUTPUT_FILE=R-${SLURM_JOB_NAME}.${SLURM_JOBID}.benchy_result.json

# bash -c is needed for the delayed interpolation of env vars to work
srun -ul --wait 60 --container-workdir $(pwd) \
--environment $(realpath env/ngc-multimeditron-24.10.toml) \
${ENROOT_ENTRYPOINT:-} bash -c "\
    set -x
    hostname
    RANK=\$SLURM_PROCID \
    WORLD_SIZE=\$SLURM_NTASKS \
    LOCAL_RANK=\$SLURM_LOCALID \
    LOCAL_WORLD_SIZE=\$SLURM_NTASKS_PER_NODE \
    \${PROFILE_CMD:-} python train.py --config config/trainings/mock_base.yaml $@\
"
echo "END TIME: $(date)"
