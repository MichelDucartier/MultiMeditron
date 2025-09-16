from typing import Annotated, Optional
from enum import IntEnum, auto
from pydantic import BaseModel, Field
import pydantic
from multimeditron.utils import pydantic_enum

@pydantic_enum
class ModelStrategy(IntEnum):
    FSDP = auto()
    DEEPSPEED = auto()
    MEGATRON = auto()

@pydantic_enum
class RewardManager(IntEnum):
    NAIVE = auto()
    PRIME = auto()
    DAPO = auto()
    ASYNC_DAPO = auto()

@pydantic_enum
class AggregationLoss(IntEnum):
    GAE = auto()
    GRPO = auto()
    REINFORCE_PLUS_PLUS = auto()
    REINFORCE_PLUS_PLUS_BASELINE = auto()
    REMAX = auto()
    RLOO = auto()
    OPO = auto()
    GRPO_PASSK = auto()
    GPG = auto()

@pydantic_enum
class KLPenalty(IntEnum):
    KL = auto()
    ABS = auto()
    MSE = auto()
    LOW_VAR_KL = auto()
    FULL = auto()

class RayConfig(BaseModel):
    num_cpus: Annotated[int, Field(gt=0)] = Field(1, description="Number of CPUs to allocate for Ray.")
    dashboard: Optional[str] = Field(None, description="Address+Port for the Ray dashboard, if left None, dashboard is disabled. Example: '0.0.0.0:8265'")

class ModelDescriptor(BaseModel):
    """
    Configuration for the model architecture.
    """
    name: Optional[str] = Field(None, description="Name of the model architecture to use.")
    path: str = Field(None, description="Path or identifier of the pretrained model to load.")
    strategy: ModelStrategy = Field(ModelStrategy.FSDP, description="Strategy for distributed training of the model.")

    @property
    def model_name(self) -> str:
        if self.name is not None:
            return self.name
        return self.path.split("/")[-1]

class ActorDescriptor(BaseModel):
    """
    Configuration for an actor in the VERL training pipeline.
    """
    model: ModelDescriptor = Field(default_factory=ModelDescriptor, description="Configuration for the actor model.")
    max_token: Annotated[int, Field(gt=0)] = Field(2048, description="Maximum number of tokens for the actor model.")

class VerlTrainer(BaseModel):
    n_gpus_per_node: Annotated[int, Field(gt=0)] = Field(1, description="Number of GPUs to use per node.")
    n_nodes: Annotated[int, Field(gt=0)] = Field(1, description="Number of nodes to use for training.")
    
    balance_batch: bool = Field(True, description="Whether to balance the batch sizes between actor and critic.")
    total_epoch: Annotated[int, Field(gt=0)] = Field(1, description="Number of epochs to train the critic per iteration.")
    total_training_steps: Optional[Annotated[int, Field(gt=0)]] = Field(None, description="Total number of training steps for the critic. If set, it will override total_epochs.")
    save_freq: Annotated[int, Field(gt=-2)] = Field(-1, description="Frequency (in iterations) to save model checkpoints. -1 means only save at the end, 0 means never save.")

    project_name: str = Field("multimeditron_verl", description="Name of the project for logging purposes.")
    experiment_name: str = Field("default", description="Name of the experiment for logging purposes.")

    use_console_logging: bool = Field(True, description="Whether to use console logging.")
    use_wandb_logging: bool = Field(False, description="Whether to use Weights & Biases for experiment tracking.")

class KlConfiguration(BaseModel):
    use_in_reward: bool = Field(False, description="Whether to use KL divergence in the reward calculation.")
    adaptive: bool = Field(False, description="Whether to use an adaptive controller for KL penalty.")
    penalty: KLPenalty = Field(KLPenalty.KL, description="Type of KL penalty to use.")
    coef: float = Field(0.001, description="Coefficient for the KL penalty term.")
    horizon: int = Field(10000, description="Horizon value for adaptive controller (if applicable).")
    target: float = Field(0.1, description="Target KL value for adaptive controller (if applicable).")

class VerlAlgorithm(BaseModel):
    """
    Configuration for the reinforcement learning algorithm.
    """
    kl: KlConfiguration = Field(default_factory=KlConfiguration, description="Configuration for KL penalty.")
    reward_manager: RewardManager = Field(RewardManager.NAIVE, description="The reward manager to use for computing rewards.")
    aggregation_loss: AggregationLoss = Field(AggregationLoss.GAE, description="The aggregation loss method to use.")
    gamma: float = Field(1.0, description="Discount factor for future rewards.")
    lam: float = Field(1.0, description="Trade-off parameter for bias-variance in GAE.")
    norm_adv_by_std_in_grpo: bool = Field(False, description="Whether to normalize advantages by standard deviation in GRPO.")

class VerlTrainConfig(BaseModel):
    ray: RayConfig = Field(default_factory=RayConfig)
    algorithm: VerlAlgorithm = Field(default_factory=VerlAlgorithm, description="Configuration for the reinforcement learning algorithm.")
    trainer: VerlTrainer = Field(default_factory=VerlTrainer, description="Configuration for the trainer.")

    actor_rollout_ref: ActorDescriptor = Field(default_factory=ActorDescriptor, description="Configuration for the actor used in rollouts.")
