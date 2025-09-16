#!/bin/bash
# Replace <your_name> by your GASPAR

runai submit-dist pytorch \
  --name meditron-training \
  --pvc mlo-scratch:/mloscratch \
  --workers 1 \
  -g 8 \
  -e WANDB_API_KEY_FILE_AT=/mloscratch/homes/your_name/keys/wandb_key.txt \
  -e HF_API_KEY_FILE_AT=/mloscratch/homes/your_name/keys/hf_key.txt \
  -e HOME=/mloscratch/homes/your_name/MultiMeditron \
  --annotation k8s.v1.cni.cncf.io/networks=kube-system/roce \
  --image registry.rcp.epfl.ch/multimeditron/basic:latest-your_name \
  --backoff-limit 0 \
  --extended-resource rdma/rdma=1 \
  --run-as-gid 83070 \
  --large-shm \
  --node-pool h100 \
  -- ./entrypoint.sh