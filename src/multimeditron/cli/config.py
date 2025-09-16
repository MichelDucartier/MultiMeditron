from typing import Annotated, Optional, List, Union
from enum import IntEnum, auto
from pydantic import BaseModel, Field
import pydantic
from multimeditron.utils import pydantic_enum

# @pydantic_enum
# class ModelStrategy(IntEnum):
#     FSDP = auto()
#     DEEPSPEED = auto()
#     MEGATRON = auto()

# @pydantic_enum
# class RewardManager(IntEnum):
#     NAIVE = auto()
#     PRIME = auto()
#     DAPO = auto()
#     ASYNC_DAPO = auto()

# @pydantic_enum
# class AggregationLoss(IntEnum):
#     GAE = auto()
#     GRPO = auto()
#     REINFORCE_PLUS_PLUS = auto()
#     REINFORCE_PLUS_PLUS_BASELINE = auto()
#     REMAX = auto()
#     RLOO = auto()
#     OPO = auto()
#     GRPO_PASSK = auto()
#     GPG = auto()

@pydantic_enum
class WarmupStyle(IntEnum):
    CONSTANT = auto()
    COSINE = auto()

@pydantic_enum
class PolicyLossMode(IntEnum):
    VANILLA = auto()
    CLIP_COV = auto()
    KL_COV = auto()
    GPG = auto()

@pydantic_enum
class LossAggregationMode(IntEnum):
    TOKEN_MEAN = auto()
    SEQ_MEAN_TOKEN_SUM = auto()
    SEQ_MEAN_TOKEN_MEAN = auto()

@pydantic_enum
class TorchDType(IntEnum):
    FLOAT16 = auto()
    BFLOAT16 = auto()
    FLOAT32 = auto()

    def to_string(self):
        match self:
            case TorchDType.FLOAT16:
                return "fp16"
            case TorchDType.BFLOAT16:
                return "bfloat16"
            case TorchDType.FLOAT32:
                return "fp32"
            case _:
                raise NotImplementedError

@pydantic_enum
class KLLossType(IntEnum):
    KL = auto()
    ABS = auto()
    MSE = auto()
    LOW_VAR_KL = auto()
    FULL = auto()

@pydantic_enum
class WeightLoadFormat(IntEnum):
    DUMMY = auto()
    HF = auto()
    MEGATRON = auto()
    SAFETENSORS = auto()

@pydantic_enum
class TruncationMode(IntEnum):
    ERROR = auto()
    LEFT = auto()
    RIGHT = auto()
    MIDDLE = auto()

class OptimConfig(BaseModel):
    lr: float = Field(1e-5, description="Learning rate for the optimizer.")
    lr_warmup_steps_ratio: float = Field(0.0, description="Ratio of total training steps to use for learning rate warmup.")
    weight_decay: float = Field(0.01, description="Weight decay (L2 regularization) factor.")
    betas: List[float] = Field([0.9, 0.999], description="Beta coefficients for the Adam optimizer.")
    clip_grad: Optional[float] = Field(None, description="Maximum gradient norm for clipping. If None, no clipping is applied.")
    min_lr_ratio: float = Field(0.0, description="Minimum learning rate ratio for the scheduler.")
    num_cycles: float = Field(0.5, description="Number of cycles for the cosine learning rate scheduler.")
    warmup_style: WarmupStyle = Field(WarmupStyle.CONSTANT, description="Style of learning rate warmup.")

    def to_hydra_conf(self):
        return {
            "_target_": "verl.workers.config.FSDPOptimizerConfig",
            "lr": self.lr,
            "lr_warmup_steps_ratio": self.lr_warmup_steps_ratio,
            "weight_decay": self.weight_decay,
            "total_training_steps": -1,  # to be filled later
            "betas": self.betas,
            "clip_grad": self.clip_grad,
            "min_lr_ratio": self.min_lr_ratio,
            "num_cycles": self.num_cycles,
            "warmup_style": self.warmup_style.name.lower(),
        }
    
class FsdpConfig(BaseModel):
    wrap_min_num_params: Annotated[int, Field(gt=-1)] = Field(0, description="Minimum number to trigger wrapping a layer with FSDP.")
    
    param_offload: bool = Field(False, description="Whether to offload parameters to CPU (trades speed for memory)")
    optimizer_offload: bool = Field(False, description="Whether to offload optimizer states to CPU")

    fsdp_size: Optional[int] = Field(None, description="Number of GPUs in each FSDP shard group, if None auto-detect")
    model_dtype: TorchDType = Field(TorchDType.FLOAT32, description="Data type for model parameters.")
    ulysses_sequence_parallel_size: int = Field(1, description="Size of sequence parallelism for ulysses-enabled models.")
    entropy_from_logits_with_chunking: bool = Field(False, description="Whether to compute entropy from logits with chunking to save memory.")
    use_torch_compile: bool = Field(True, description="Whether to use torch.compile to optimize in FSDP.")
    entropy_checkpointing: bool = Field(False, description="Whether to use activation checkpointing for entropy computation.")
    forward_only: bool = Field(False, description="If True, only perform forward pass (no backward). Useful for inference or evaluation.")
    fsdp_version: Annotated[int, Field(gt=0, lt=3)] = Field(2, description="Version of FSDP to use. 1 for FSDP1, 2 for FSDP2.")

    fsdp1_forward_prefetch: bool = Field(False, description="Only for FSDP1: prefetch the next forward-pass all-gather before the current forward computation")
    fsdp1_use_orig_params: bool = Field(False, description="Only for FSDP1: use original parameters in FSDP")

    fsdp2_offload_policy: bool = Field(False, description="Only for FSDP2: offload param/grad/optimizer during train")
    fsdp2_reshard_after_forward: bool = Field(True, description="Only for FSDP2: Reshard after forward pass to save memory")

    def to_hydra_conf(self):
        return {
            "_target_": "verl.workers.config.FSDPConfig",
            "wrap_policy": {
                "min_num_params": self.wrap_min_num_params
            },
            "param_offload": self.param_offload,
            "optimizer_offload": self.optimizer_offload,
            "offload_policy": self.fsdp2_offload_policy,
            "reshard_after_forward": self.fsdp2_reshard_after_forward,
            "fsdp_size": self.fsdp_size if self.fsdp_size is not None else -1,
            "forward_prefetch": self.fsdp1_forward_prefetch,
            "model_dtype": self.model_dtype.to_string(),
            "ulysses_sequence_parallel_size": self.ulysses_sequence_parallel_size,
            "entropy_from_logits_with_chunking": self.entropy_from_logits_with_chunking,
            "use_torch_compile": self.use_torch_compile,
            "entropy_checkpointing": self.entropy_checkpointing,
            "forward_only": self.forward_only,
            "strategy": "fsdp" if self.fsdp_version == 1 else "fsdp2",
        }

class PolicyLossConfig(BaseModel):
    loss_mode: PolicyLossMode = Field(PolicyLossMode.VANILLA, description="Mode of policy loss to use.")
    clip_cov_ratio: float = Field(0.0002, description="Ratio of tokens to be clipped for clip-cov loss.")
    clip_cov_lb: float = Field(1.0, description="Lower bound for clip-cov loss.")
    clip_cov_ub: float = Field(1.0, description="Upper bound for clip-cov loss.")
    kl_cov_ratio: float = Field(0.0002, description="Ratio of tokens to be clipped for kl-cov loss.")
    ppo_kl_coef: float = Field(0.1, description="Coefficient for KL penalty")

    def to_hydra_conf(self):
        return {
            "_target_": "verl.workers.config.PolicyLossConfig",
            "loss_mode": str(self.loss_mode),
            "clip_cov_ratio": self.clip_cov_ratio,
            "clip_cov_lb": self.clip_cov_lb,
            "clip_cov_ub": self.clip_cov_ub,
            "kl_cov_ratio": self.kl_cov_ratio,
            "ppo_kl_coef": self.ppo_kl_coef,
        }

class CheckpointConfig(BaseModel):
    save_contents: List[str] = Field(["model", "optimizer", "extra"], description="List of contents to save in the checkpoint. Options include 'model', 'optimizer', 'extra', 'hf_model'.")
    load_contents: Optional[List[str]] = Field(None, description="List of contents to load from the checkpoint. Options include 'model', 'optimizer', 'extra', 'hf_model'. If None, load same as save_contents.")
    async_save: bool = Field(False, description="Whether to save checkpoints asynchronously.")

    def to_hydra_conf(self):
        return {
            "_target_": "verl.trainer.config.CheckpointConfig",
            "save_contents": self.save_contents,
            "load_contents": self.load_contents if self.load_contents is not None else self.save_contents,
            "async_save": self.async_save,
        }

class ProfilerConfig(BaseModel):
    tool: Optional[str] = Field(None, description="Profiler tool to use, e.g., 'nsys', 'npu', 'torch'. If None, profiling is disabled.")
    enable: bool = Field(False, description="Whether to enable profiling.")
    all_ranks: bool = Field(False, description="Whether to profile all ranks")
    ranks: List[int] = Field([], description="List of ranks to profile if all_ranks is False.")
    save_path: Optional[str] = Field(None, description="Path to save profiler output. If None, a default path will be used.")

    nsys_discrete: bool = Field(True, description="Whether to use discrete mode in nsys profiler.")

    npu_contents: List[str] = Field([], description="List of contents to collect in npu profiler. Options: 'npu', 'cpu', 'memory', 'shapes', 'module', 'stack'.")
    npu_level: Optional[int] = Field(None, description="Level of detail for npu profiler (0-3). If None, default level is used.")
    npu_analysis: bool = Field(True, description="Whether to automatically parse the data")
    npu_discrete: bool = Field(True, description="Whether to use discrete mode in npu profiler.")

    torch_step_start: int = Field(0, description="Step to start profiling in torch profiler.")
    torch_step_end: Optional[int] = Field(None, description="Stop profile mini-batch in training")

    torch_memory_trace_alloc_max_entries: int = Field(1_000_000, description="Max number of memory allocation records to keep in torch profiler.")
    torch_memory_stack_depth: int = Field(32, description="Stack depth for memory profiling in torch profiler.")

    def to_hydra_conf(self):
        return {
            "_target_": "verl.utils.profiler.ProfilerConfig",
            "tool": self.tool,
            "enable": self.enable,
            "all_ranks": self.all_ranks,
            "ranks": self.ranks,
            "save_path": self.save_path,
            "tool_config": {
                "nsys": {
                    "_target_": "verl.utils.profiler.config.NsightToolConfig",
                    "discrete": self.nsys_discrete,
                },
                "npu": {
                    "_target_": "verl.utils.profiler.config.NPUToolConfig",
                    "contents": self.npu_contents,
                    "level": "level_node" if self.npu_level is None else f"level{self.npu_level}",
                    "analysis": self.npu_analysis,
                    "discrete": self.npu_discrete,
                },
                "torch": {
                    "_target_": "verl.utils.profiler.config.TorchProfilerToolConfig",
                    "step_start": self.torch_step_start,
                    "step_end": self.torch_step_end,
                },
                "torch_memory": {
                    "_target_": "verl.utils.profiler.config.TorchMemoryToolConfig",
                    "trace_alloc_max_entries": self.torch_memory_trace_alloc_max_entries,
                    "stack_depth": self.torch_memory_stack_depth,
                }
            }

        }

class ActorConfig(BaseModel):
    optim: OptimConfig = Field(default_factory=OptimConfig, description="Optimizer configuration for the actor.")
    fsdp: FsdpConfig = Field(default_factory=FsdpConfig, description="FSDP configuration for the actor.")
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig, description="Configuration for saving and loading checkpoints.")
    policy_loss: PolicyLossConfig = Field(default_factory=PolicyLossConfig, description="Configuration for policy loss.")
    profiler: ProfilerConfig = Field(default_factory=ProfilerConfig, description="Profiler configuration for the actor.")
    
    ppo_mini_batch_size: Annotated[int, Field(gt=0)] = Field(256, description="Split each sample into sub-batches of this size for PPO training.")
    use_dynamic_bsz: bool = Field(False, description="Whether to automatically adjust batch size based on GPU memory.")
    ppo_max_token_len_per_gpu: Annotated[int, Field(gt=0)] = Field(16384, description="Maximum token length per GPU for PPO training. Typically it should be: n * data.max_prompt_length + data.max_response_length.")
    clip_ratio: float = Field(0.2, description="Clipping ratio for PPO.")
    clip_ratio_low: float = Field(0.2, description="Lower bound for asymmetric clipping ratio in PPO (used in dual clip PPO).")
    clip_ratio_high: float = Field(0.2, description="Upper bound for asymmetric clipping ratio in PPO (used in dual clip PPO).")
    # freeze_vision_tower: bool = Field(False, description="Whether to freeze the vision tower during training.")

    clip_ratio_c: float = Field(3.0, description="Constant C for Dual-clip PPO; clips when advantage < 0 and ratio > C")
    loss_agg_mode: LossAggregationMode = Field(LossAggregationMode.TOKEN_MEAN, description="Method to aggregate loss over tokens and sequences.")
    entropy_coef: float = Field(0.0, description="Entropy regularization coefficient in PPO loss.")
    tis_imp_ratio_cap: float = Field(-1, description="Truncated Importance Sampling ratio cap. If < 0, no truncation is applied.")
    use_kl_loss: bool = Field(False, description="Whether to use KL divergence loss between current policy and reference policy. True for GRPO.")
    use_torch_compile: bool = Field(True, description="Whether to use torch.compile to optimize the model.")
    kl_loss_coef: float = Field(0.001, description="KL loss coefficient when use_kl_loss is True.")
    kl_loss_type: KLLossType = Field(KLLossType.KL, description="Type of KL penalty to use.")
    ppo_epochs: Annotated[int, Field(gt=0)] = Field(1, description="Number of epochs to train the actor per batch.")
    shufle: bool = Field(False, description="Whether to shuffle training data across PPO epochs.")
    use_fused_kernels: bool = Field(True, description="Whether to use custom fused kernels (e.g., FlashAttention, fused MLP) for better speed and memory.")
    

    def to_hydra_conf(self):
        return {
            "_target_": "verl.workers.config.ActorConfig",
            "strategy": "fsdp" if self.fsdp.fsdp_version == 1 else "fsdp2",
            "ppo_mini_batch_size": self.ppo_mini_batch_size,
            "ppo_micro_batch_size": None, # to be filled later
            "ppo_micro_batch_size_per_gpu": None, # to be filled later
            "use_dynamic_bsz": self.use_dynamic_bsz,
            "ppo_max_token_len_per_gpu": self.ppo_max_token_len_per_gpu,
            "clip_ratio": self.clip_ratio,
            "clip_ratio_low": self.clip_ratio_low,
            "clip_ratio_high": self.clip_ratio_high,
            "freeze_vision_tower": False, # TODO
            "policy_loss": self.policy_loss.to_hydra_conf(),
            "clip_ratio_c": self.clip_ratio_c,
            "loss_agg_mode": str(self.loss_agg_mode),
            "entropy_coef": self.entropy_coef,
            "tis_imp_ratio_cap": -1 if self.tis_imp_ratio_cap < 0 else self.tis_imp_ratio_cap,
            "use_kl_loss": self.use_kl_loss,
            "use_torch_compile": self.use_torch_compile,
            "kl_loss_coef": self.kl_loss_coef,
            "kl_loss_type": str(self.kl_loss_type),
            "ppo_epochs": self.ppo_epochs,
            "shuffle": self.shufle,
            "checkpoint": self.checkpoint.to_hydra_conf(),
            "optim": self.optim.to_hydra_conf(),
            "used_fused_kernels": self.use_fused_kernels,
            "profiler": self.profiler.to_hydra_conf(),
        }

class RolloutConfig(BaseModel):
    name: str = Field("???", description="hf/path or local path to the model")
    enable_async: bool = Field(False, description="Whether to enable asynchronous rollout.")
    temperature: float = Field(1.0, description="Sampling temperature for the rollout model.")
    top_k: Optional[int] = Field(None, description="Top-k sampling for the rollout model. If None, no top-k filtering is applied.")
    top_p: float = Field(1.0, description="Top-p (nucleus) sampling for the rollout model.")
    prompt_length: Annotated[int, Field(gt=0)] = Field(512, description="Maximum prompt length")
    response_length: Annotated[int, Field(gt=0)] = Field(1024, description="Maximum response length")
    dtype: TorchDType = Field(TorchDType.BFLOAT16, description="Data type for model parameters during rollout.")
    gpu_memory_utilization: Annotated[float, Field(gt=0.0, lt=1.0)] = Field(0.5, description="Target GPU memory utilization used by VLLM/SGLang for KV cache allocation.")
    ignore_eos: bool = Field(False, description="Whether to ignore EOF and continue generation until reaching max length.")
    enforce_eager: bool = Field(False, description="Whether to enforce eager generation (disable CUDA graph). Tradeoff throughput/latency.")
    vllm_cudagraph_capture_sizes: Optional[List[int]] = Field(None, description="List of token lengths to capture CUDA graphs for vLLM. Require enforce_eager=False. You can use smaller batch size to save memory in cuda graph.")
    free_cache_engine: bool = Field(True, description="Whether to free engine KVCache after generation")
    tensor_model_parallel_size: Annotated[int, Field(gt=0)] = Field(2, description="Tensor model parallel size for the rollout model, not effective for HF")
    max_num_batched_tokens: Annotated[int, Field(gt=0)] = Field(8192, description="Maximum number of tokens to batch together.")
    max_model_len: Optional[int] = Field(None, description="Maximum length for rollout")
    max_num_seqs: Optional[int] = Field(1024, description="Maximum length of sequences")
    enable_chunked_prefill: bool = Field(True, description="May get higher throughput when set to True. When activated, please increase max_num_batched_tokens or decrease max_model_len.")
    enable_prefix_caching: bool = Field(True, description="Prefix caching kv-cache blocks is a popular optimization in LLM inference to avoid redundant prompt computation.")
    load_format: WeightLoadFormat = Field(WeightLoadFormat.HF, description="Format to load the model weights.")
    do_sample: bool = Field(True, description="Whether to use sampling (True) or greedy decoding (False) during generation.")
    n: int = Field(1, description="Number of responses to generate per prompt. > 1 for GRPO")
    over_sample_rate: float = Field(0.0, description="Over-sample rate parameter controls the early termination threshold for training rollouts, where the system will abort remaining requests when (1 - over_sample_rate) * total_requests have been completed.")
    sglang_multi_stage_wake_up: bool = Field(False, description="Whether to wake up inference engine in multi-stage for SGLang.")
    
    vllm_kwargs: dict = Field(default_factory=dict, description="Additional keyword arguments to pass to vLLM engine.")
    sglang_kwargs: dict = Field(default_factory=dict, description="Additional keyword arguments to pass to SGLang engine.")

    update_weights_bucket_megabytes: Annotated[int, Field(gt=0)] = Field(512, description="Specifies the tensor bucket size (in megabytes) for batch weight updates during rollout operations. This controls the maximum payload size for a signle weight update request. For best performance it is recommended to enable `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES`, manually set `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` when using tensor parallelism >= 8.")

    profiler: ProfilerConfig = Field(default_factory=ProfilerConfig, description="Profiler configuration for the rollout model.")

    def to_hydra_conf(self,
                      log_prob_use_dynamic_bsz: bool = False,
                      ppo_max_token_len_per_gpu: int = 16384):
        return {
            "_target_": "verl.workers.config.RolloutConfig",
            "name": self.name,
            "mode": "async" if self.enable_async else "sync",
            "temperature": self.temperature,
            "top_k": -1 if self.top_k is None else self.top_k,
            "top_p": self.top_p,
            "prompt_length": self.prompt_length,
            "response_length": self.response_length,
            "dtype": self.dtype.to_string(),
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "ignore_eos": self.ignore_eos,
            "enforce_eager": self.enforce_eager,
            "cudagraph_capture_sizes": self.vllm_cudagraph_capture_sizes,
            "free_cache_engine": self.free_cache_engine,
            "tensor_model_parallel_size": self.tensor_model_parallel_size,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "enable_chunked_prefill": self.enable_chunked_prefill,
            "enable_prefix_caching": self.enable_prefix_caching,
            "load_format": self.load_format.name.lower(),
            "log_prob_micro_batch_size": None, # to be filled later
            "log_prob_micro_batch_size_per_gpu": None, # to be filled later
            "log_prob_use_dynamic_bsz": log_prob_use_dynamic_bsz, # to be filled later
            "log_prob_max_token_len_per_gpu": ppo_max_token_len_per_gpu, # to be filled later
            "disable_log_stats": True,
            "do_sample": self.do_sample,
            "n": self.n,
            "over_sample_rate": self.over_sample_rate,
            "multi_stage_wake_up": self.sglang_multi_stage_wake_up,
            "engine_kwargs": {
                "vllm": self.vllm_kwargs,
                "sglang": self.sglang_kwargs,
            },
            "val_kwargs": {
                "_target_": "verl.workers.config.SamplingConfig",
                "temperature": 0,
                "top_k": -1,
                "top_p": 1.0,
                "n": 1,
                "do_sample": False,
            },
            "multi_turn": {
                "_target_": "verl.workers.config.MultiTurnConfig",
                "enable": False,
            },
            "calculate_log_probs": False,
            "update_weights_bucket_megabytes": self.update_weights_bucket_megabytes,
            "trace": {
                "_target_": "verl.workers.config.TraceConfig",
                "backend": None,
                "token2text": False,
            },
            "skip_rollout": False,
            "skip_dump_dir": None,
            "skip_tokenizer_init": True, # Token in, token out for generation
            "profiler": self.profiler.to_hydra_conf(),
        }

class HfConfig(BaseModel):
    path: str = Field("???", description="Path to the HuggingFace repository or local directory.")
    use_shm: bool = Field(False, description="Whether to use shared memory for loading model weights.")
    
    def to_hydra_conf(self, trust_remote_code: bool = False):
        return {
            "_target_": "verl.workers.config.HFModelConfig",
            "path": self.path,
            "hf_config_path": None,
            "tokenizer_path": None,
            "use_shm": self.use_shm,
            "trust_remote_code": trust_remote_code,
            "custom_chat_template": None,
            "external_lib": None,
            "override_config": {},
            "enable_gradient_checkpointing": True,
            "enable_activation_offload": False,
            "use_remove_padding": False,
            "lora_rank": 0,
            "lora_alpha": 16,
            "target_modules": "all-linear",
            "exclude_modules": None,
            "use_liger": False,
            "use_fused_kernels": False,
            "fused_kernel_options": {
                "impl_backend": "torch",
            }
        }

class DataConfig(BaseModel):
    tokenizer: Optional[str] = Field(None, description="Path to the tokenizer. If None, use the model's default tokenizer.")
    use_shm: bool = Field(False, description="Whether to use shared memory for data loading")
    train_files: Union[List[str], str] = Field(None, description="Training set parquet. Can be a list of a single file. Read everything in memory, cannot be too large.")
    val_files: Optional[Union[List[str], str]] = Field(None, description="Validation set parquet. Can be a list of a single file. Read everything in memory, cannot be too large.")
    reward_fn_key: str = Field("data_source", description="The field used to select the reward function (if using different ones per example).")
    train_batch_size: Annotated[int, Field(gt=0)] = Field(1024, description="Batch size for one iteration of different RL algorithms.")
    val_batch_size: Optional[Annotated[int, Field(gt=0)]] = Field(None, description="Batch size for validation")

    shuffle: bool = Field(True, description="Whether to shuffle the training data.")
    validation_shuffle: bool = Field(False, description="Whether to shuffle the validation data.")
    dataloader_num_workers: Annotated[int, Field(gt=0)] = Field(5, description="Number of worker processes for data loading.")

    max_prompt_length: Annotated[int, Field(gt=0)] = Field(512, description="Maximum length of the prompt (in tokens). All prompts will be left-padded to this length, check truncation mode for handling overlong prompts.")
    max_response_length: Annotated[int, Field(gt=0)] = Field(1024, description="Maximum length of the response (in tokens), rollout in RL algorithms generates up to this length.")
    filter_overlong_prompts: bool = Field(False, description="Whether to filter out prompts that exceed max_prompt_length.")
    truncation: TruncationMode = Field(TruncationMode.ERROR, description="Truncation mode for prompts that exceed max_prompt_length.")

    prompt_key: str = Field("prompt", description="The field in the dataset where the prompt is located. Default is 'prompt'.")
    image_key: str = Field("images", description="The field in the dataset where the image is located. Default is 'image'.")
    video_key: str = Field("videos", description="The field in the dataset where the video is located. Default is 'video'.")

    def to_hydra_conf(self):
        return {
            "tokenizer": self.tokenizer,
            "use_shm": self.use_shm,
            "train_files": self.train_files,
            "val_files": self.val_files,
            "prompt_key": self.prompt_key,
            "reward_fn_key": self.reward_fn_key,
            "max_prompt_length": self.max_prompt_length,
            "max_response_length": self.max_response_length,
            "train_batch_size": self.train_batch_size,
            "val_batch_size": self.val_batch_size,
            "return_raw_input_ids": False,
            "return_raw_chat": False,
            "return_full_prompt": False,
            "shuffle": self.shuffle,
            "dataloader_num_workers": self.dataloader_num_workers,
            "validation_shuffle": self.validation_shuffle,
            "filter_overlong_prompts": self.filter_overlong_prompts,
            "truncation": str(self.truncation),
            "image_key": self.image_key,
            "video_key": self.video_key,
            "trust_remote_code": False,
            "custom_cls": {
                "path": None,
                "name": None,
            },
            "return_multi_modal_inputs": True,
            "sampler": {
                "class_path": None,
                "class_name": None,
            },
            "datagen": {
                "path": None,
                "name": None,
            },
            "apply_chat_template_kwargs": {},
        }


class ActorRolloutRefConfig(ActorConfig):
    actor: ActorConfig = Field(default_factory=ActorConfig, description="Configuration for the actor model.")
    rollout: RolloutConfig = Field(default_factory=RolloutConfig, description="Configuration for the rollout model.")
    model: HfConfig = Field(default_factory=HfConfig, description="Configuration for loading models from HuggingFace or local path.")

    def to_hydra_conf(self):
        return {
            "actor": self.actor.to_hydra_conf(),
            "ref": None,
            "rollout": self.rollout.to_hydra_conf(
                log_prob_use_dynamic_bsz=self.actor.use_dynamic_bsz,
                ppo_max_token_len_per_gpu=self.actor.ppo_max_token_len_per_gpu,
            ),
            "model": self.model.to_hydra_conf(),
        }
    
class VerlConfig(BaseModel):
    actor_rollout_ref: ActorRolloutRefConfig = Field(default_factory=ActorRolloutRefConfig, description="Configuration for the actor, rollout, and reference models.")
    data: DataConfig = Field(default_factory=DataConfig, description="Configuration for the dataset.")

    def to_hydra_conf(self):
        return {
            "actor_rollout_ref": self.actor_rollout_ref.to_hydra_conf(),
            "data": self.data.to_hydra_conf(),
        }
