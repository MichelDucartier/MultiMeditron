import ray
import random
from multimeditron.cli.config import VerlConfig
from pprint import pprint

@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, cfg: VerlConfig, trust_remote_code: bool = False):
        from transformers import AutoTokenizer

        pprint(cfg.model_dump())

        # Instantiate tokenizer
        print("Instantiating tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.actor_rollout_ref.model.path,
            use_fast=True,
            trust_remote_code=trust_remote_code
        )
        
        # For each strategy, we would have different training loops
        # assert cfg.actor_rollout_ref.model.strategy == cfg.critic.model.strategy, "Currently only support same strategy for actor and critic"

        match cfg.actor_rollout_ref.model.strategy:
            case ModelStrategy.FSDP:
                from verl.single_controller.ray import RayWorkerGroup
                from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker

                ray_worker_group_cls = RayWorkerGroup
            case ModelStrategy.MEGATRON:
                from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
                from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker

                ray_worker_group_cls = NVMegatronRayWorkerGroup
            case _:
                print(f"Strategy {cfg.actor_rollout_ref.model.strategy} not implemented yet.")
                raise NotImplementedError

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [cfg.trainer.n_gpus_per_node] * cfg.trainer.n_nodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }

        # We should adopt a multi-source reward function here
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # - finally, we combine all the rewards together
        # - The reward type depends on the tag of the data

        # If we want reward model
        # if config.reward_model.enable:
        #     if config.reward_model.strategy == "fsdp":
        #         from verl.workers.fsdp_workers import RewardModelWorker
        #     elif config.reward_model.strategy == "megatron":
        #         from verl.workers.megatron_workers import RewardModelWorker
        #     else:
        #         raise NotImplementedError
        #     role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        #     mapping[Role.RewardModel] = global_pool_id
        
        # Reference model
        # if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        if cfg.algorithm.kl.use_in_reward:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # reward_manager_name = config.reward_model.get("reward_manager", "naive")
        match cfg.algorithm.reward_manager:
            case RewardManager.NAIVE:
                from verl.workers.reward_manager import NaiveRewardManager
                reward_manager_cls = NaiveRewardManager
            case RewardManager.PRIME:
                from verl.workers.reward_manager import PrimeRewardManager
                reward_manager_cls = PrimeRewardManager
            case RewardManager.DAPO:
                from verl.workers.reward_manager import DAPORewardManager
                reward_manager_cls = DAPORewardManager
            case RewardManager.ASYNC_DAPO:
                raise NotImplementedError
            case _:
                raise NotImplementedError

        # compute_score = get_custom_reward_fn(config)
        reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=0,
            compute_score=compute_score,
            # reward_fn_key=cfg.data.reward_fn_key,
            # max_resp_len=cfg.actor_rollout_ref.max_token,
            # overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )

        # Note that we always use function-based RM for validation
        val_reward_fn = reward_manager_cls(
            tokenizer=tokenizer,
            num_examine=1,
            compute_score=compute_score,
            # reward_fn_key=config.data.reward_fn_key,
            # max_resp_len=cfg.actor_rollout_ref.max_token,
            # overlong_buffer_cfg=config.reward_model.overlong_buffer,
        )
        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        trainer = RayPPOTrainer(
            config={
            },
            tokenizer=tokenizer,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=...,
            val_dataset=...,
            # collate_fn=...,
            train_sampler=None,
            # device_name="cuda",
        )
        trainer.init_workers()
        trainer.fit()

def compute_score(data_source, solution_str, ground_truth, extra_info):
    if extra_info is None or "question" not in extra_info or "url" not in extra_info:
        raise ValueError("Extra info is required and must contain 'question' and 'url'")
    
    do_print = False
    if random.randint(0, 512) == 1:  
        do_print = True
    if do_print:
        print(f"Response Case: {solution_str}, Question: {extra_info['question']}, GT: {ground_truth}")

    response = solution_str
    response_lower = response.lower()
    score = response_lower.count("a") / len(response_lower) if len(response_lower) > 0 else 0

    return {
        "score": score,
        "acc": 0.0,
        "pred": "Maybe",
    }