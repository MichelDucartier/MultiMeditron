from multimeditron.cli.config import VerlTrainConfig, ModelStrategy, RewardManager
from omegaconf import OmegaConf

def make_cfg(cfg: VerlTrainConfig):
    log_backend = []
    if cfg.trainer.use_console_logging:
        log_backend.append("console")
    if cfg.trainer.use_wandb_logging:
        log_backend.append("wandb")

    d = {
        "actor_rollout_ref": {
            "hybrid_engine": True,
            "nccl_timeout": 600,
            "rollout": {
                # May get higher throughput when set to True. When activated, Please increase max_num_batched_tokens or decrease max_model_len.
                "enable_chunked_prefill": True,

                # Prefix caching kv-cache blocks is a popular optimization in LLM inference to avoid redundant prompt computations.
                "enable_prefix_caching": True,

                # Which loader to use for rollout model weights: dummy_dtensor, hf, megatron, etc.
                # safetensors (for huge model, and set use_shm=True); dummy_dtensor: randomly init model weight
                "load_format": "dummy_dtensor",

                # For huge model, layered summon can save memory (prevent OOM) but make it slower             
                "layered_summon": False,
            }
        },
        "custom_reward_function": {
            # The path to the file containing your customized reward function.
            # If not specified, pre-implemented reward functions will be used.
            "path": None,

            # The name of the reward function within the specified file. Default is 'compute_score'.
            "name": "compute_score",
        },
        "algorithm": {
            "_target_": "verl.trainer.config.AlgoConfig",
            "gamma": cfg.algorithm.gamma,
            "lam": cfg.algorithm.lam,
            "adv_estimator": cfg.algorithm.aggregation_loss.name.lower(),
            "norm_adv_by_std_in_grpo": cfg.algorithm.norm_adv_by_std_in_grpo,
            "use_kl_in_reward": cfg.algorithm.kl.use_in_reward,
            "kl_penalty": cfg.algorithm.kl.penalty.name.lower(),
            "kl_ctrl": {
                # Required when using verl.utils.omega_conf_to_dataclass to instantiate dataclass configs
                "_target_": "verl.trainer.config.KLControlConfig",

                # KL control type: "fixed" or "adaptive"
                "type": "adaptive" if cfg.algorithm.kl.adaptive else "fixed",

                # Initial coefficient for KL penalty
                "kl_coef": cfg.algorithm.kl.coef,

                # Horizon value for adaptive controller (if enabled)
                "horizon": cfg.algorithm.kl.horizon,

                # Target KL divergence (used for adaptive controller)
                "target_kl": cfg.algorithm.kl.target,
            },
            "use_pf_ppo": cfg.algorithm.use_pf_ppo,
            "pf_ppo": {
                # Method for reweighting samples: "pow", "max_min", or "max_random"
                "reweight_method": "pow",

                # Power used for weight scaling in "pow" method
                "weight_pow": 2.0,
            },
        },
        "trainer": {
            # Whether to balance batch sizes across workers
            "balance_batch": cfg.trainer.balance_batch,

            # Number of epochs in training
            "total_epochs": cfg.trainer.total_epoch,

            # Total training steps (can be set explicitly)
            "total_training_steps": cfg.trainer.total_training_steps,

            # Project name for experiment tracking
            "project_name": cfg.trainer.project_name,

            # Experiment name for run identification in tracking tools
            "experiment_name": cfg.trainer.experiment_name,

            # Logging backends to use: "console", "wandb", "tensorboard", etc.
            "log_backend": log_backend,

            # Number of generations to log during validation
            "log_val_generations": 0,

            # Directory for logging rollout data; no dump if null
            "rollout_data_dir": None,

            # Directory for logging rollout data; no dump if null
            "rollout_data_dir": None,

            # Directory for logging validation data; no dump if null
            "validation_data_dir": None,

            # Number of nodes used in the training
            "nnodes": cfg.trainer.n_nodes,

            # Number of GPUs per node
            "n_gpus_per_node": cfg.trainer.n_gpus_per_node,

            # Safe frequency (by iteration) for model checkpointing
            "save_freq": cfg.trainer.save_freq,

            # ESI refers to the elastic server instance used during training, similar to the training plan. For example,
            # if you purchase 10 hours of computing power, the ESI will automatically shut down after 10 hours of training.
            # To ensure a checkpoint is saved before ESI shuts down, the system will start saving a checkpoint in advance.
            # The advance time is calculated as: Advance Time = Longest historical step duration + Checkpoint save duration + esi_redundant_time.
            # Here, esi_redundant_time is a user-defined value that further extends the advance time for added safety.
            "esi_redundant_time": 0,

            # Resume mode: "auto", "disable", or "resume_path"
            # "auto": resume from last checkpoint if available
            # "disable": start from scratch
            # "resume_path": resume from a user-defined path
            "resume_mode": "auto",

            # Path to resume training from (only used when resume_mode is "resume_path")
            "resume_from_path": None,

            # Whether to run validation before training begins
            "val_before_train": True,

            # Whether to run validation only
            "val_only": False,

            # Validation frequency (in training iterations)
            "test_freq": -1,

            # Number of iterations to warm up the critic before updating policy
            "critic_warmup": 0,

            # Default path to distributed filesystem for saving checkpoints
            "default_hdfs_dir": None,

            # Whether to delete local checkpoints after loading
            "del_local_ckpt_after_load": False,

            # Default local directory for saving checkpoints
            "default_local_dir": "checkpoints/${trainer.project_name}/${trainer.experiment_name}",

            # Maximum number of actor checkpoints to keep
            "max_actor_ckpt_to_keep": None,

            # Maximum number of critic checkpoints to keep
            "max_critic_ckpt_to_keep": None,

            # Timeout (in seconds) for Ray worker to wait for registration
            "ray_wait_register_center_timeout": 300,

            # Device to run training on (e.g., "cuda", "cpu")
            "device": "cuda",

            # whether to use legacy worker implementation
            #  mode: "auto", "enable", or "disable"
            "use_legacy_worker_impl": "auto",
        },
        "global_profile": {
            "_target_": "verl.utils.profiler.ProfilerConfig",

            # Profiling tool: choose between nsys, npu, torch, torch_memory
            "tool": None,

            # profile steps
            "steps": None,

            # Whether to combine continuous steps into one database.
            ## If True, worker.profiler.discrete must be False, [1,2] in one, [5] in another.
            ## If False, [1] in one, [2] in another, [5] in another.
            "profile_continuous_steps": False,

            # Path to save profiling contents
            "save_path": "outputs/profile",

            # Specific tool configs, can use +profiler.tool_config.[tool].xxx to config
            "global_tool_config": {
                # Required when using verl.utils.omega_conf_to_dataclass to instantiate dataclass configs
                "_target_": "verl.utils.profiler.config.NsightToolConfig",

                # nsys config
                "nsys": {},

                # True for each task has its own database, False for all tasks in one training step share one database.
                "discrete": False,

                # controller Nvidia Nsight Systems Options. Must set when profile_steps is not None.
                ## reference https://docs.nvidia.com/nsight-systems/UserGuide/index.html
                ## reference https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html
                "controller_nsight_options": {
                    # Select the API(s) to be traced.
                    "trace": "cuda,nvtx,cublas,ucx",

                    # Track the GPU memory usage by CUDA kernels. Must be string type "true" or "false".
                    "cuda-memory-usage": "true",

                    # CUDA graphs will be traced as a whole
                    "cuda-graph-trace": "graph",

                },

                # worker Nvidia Nsight Systems Options. Must set when profile_steps is not None.
                "worker_nsight_options": {
                    # Select the API(s) to be traced.
                    "trace": "cuda,nvtx,cublas,ucx",

                    # Track the GPU memory usage by CUDA kernels. Must be string type "true" or "false".
                    "cuda-memory-usage": "true",

                    # CUDA graphs will be traced as a whole
                    "cuda-graph-trace": "graph",

                    # Profiling only in a range of torch.cuda.profiler.start and stop. Do not change this config.
                    "capture-range": "cudaProfilerApi",

                    # Specify the desired behavior when a capture range ends.
                    # In verl we need the torch.cuda.profiler.start/stop pair to repeats n times.
                    # valid values are "repeat-shutdown:n" or null.
                    # For normal whole step profiling, n = len(profile_steps);
                    # but for discrete profiling, n = len(profile_steps) * Number(subtasks).
                    # Or you can just leave it null and the program will use n = len(profile_steps) * 6;
                    "capture-range-end": None,

                    # Send signal to the target application's process group. We let the program to exit by itself.
                    "kill": None,
                },

                # enable memory visualization for debugging memory usage
                "torch_memory": {},

                #  Maximum number of allocation entries to record
                "trace_alloc_max_entries": 100_000,

                # The depth of the call stack to capture for each allocation
                "stack_depth": 32,

                # 'alloc': records only allocation events || 'state': records memory state changes || 'all': records both.
                "context": "all",

                # 'python': records Python stacks || 'cpp': records C++ stacks (available in some versions) || 'all': records both.
                "stacks": "all",

                # devices, record_context etc.
                "kw_args": {},
            },
            # configs related to ray
            "ray_kwargs": {
                # configs related to ray initialization
                "ray_init": {},

                # Number of CPUs for Ray. Use a fixed number instead of null when using SLURM.
                "num_cpus": None,

                # Path to save Ray timeline JSON for performance profiling
                "timeline_json_file": None
            },
        }
    }

    return OmegaConf.to_container(d)