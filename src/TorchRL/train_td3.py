# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""TD3 Example.

This is a simple self-contained example of a TD3 training script.

It supports state environments like MuJoCo.

The helper functions are coded in the utils.py associated with this script.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.cuda
import tqdm
from tensordict.nn import CudaGraphModule
from torchrl._utils import compile_with_warmup, timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import generate_exp_name, get_logger
from utils_td3 import (
    dump_video,
    log_metrics,
    make_collector,
    make_environment,
    make_loss_module,
    make_optimizer,
    make_replay_buffer,
    make_td3_agent,
)

torch.set_float32_matmul_precision("high")


def main(cfg: DictConfig):  # noqa: F821
    device = cfg.network.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    # Create logger
    logger = None
    if cfg.logger.backend:
        exp_name = f"{cfg.lastfolder}_{cfg.rep}"
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name=f"results/td3_{cfg.env.name}_{cfg.collector.total_frames}/{cfg.hps}/",
            experiment_name=exp_name,
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    # Set seeds
    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    # Create environments
    train_env, eval_env = make_environment(cfg, logger=logger, device=device)

    # Create agent
    model, exploration_policy = make_td3_agent(cfg, train_env, eval_env, device)

    # Create TD3 loss
    loss_module, target_net_updater = make_loss_module(cfg, model)

    compile_mode = None
    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode
        if compile_mode in ("", None):
            if cfg.compile.cudagraphs:
                compile_mode = "default"
            else:
                compile_mode = "reduce-overhead"

    # Create off-policy collector
    collector = make_collector(
        cfg,
        train_env,
        exploration_policy,
        compile_mode=compile_mode,
        device=device,
    )

    # Create replay buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        scratch_dir=cfg.replay_buffer.scratch_dir,
        device=device,
        compile=bool(compile_mode),
    )

    # Create optimizers
    optimizer_actor, optimizer_critic = make_optimizer(cfg, loss_module)

    prb = cfg.replay_buffer.prb

    def update(sampled_tensordict, update_actor, prb=prb):

        # Compute loss
        q_loss, *_ = loss_module.value_loss(sampled_tensordict)

        # Update critic
        q_loss.backward()
        optimizer_critic.step()
        optimizer_critic.zero_grad(set_to_none=True)

        # Update actor
        if update_actor:
            actor_loss, *_ = loss_module.actor_loss(sampled_tensordict)

            actor_loss.backward()
            optimizer_actor.step()
            optimizer_actor.zero_grad(set_to_none=True)

            # Update target params
            target_net_updater.step()
        else:
            actor_loss = q_loss.new_zeros(())

        return q_loss.detach(), actor_loss.detach()

    if cfg.compile.compile:
        update = compile_with_warmup(update, mode=compile_mode, warmup=1)

    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            category=UserWarning,
        )
        update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=5)

    # Main loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    init_random_frames = cfg.collector.init_random_frames
    num_updates = int(cfg.collector.frames_per_batch * cfg.optim.utd_ratio)
    delayed_updates = cfg.optim.policy_update_delay
    eval_rollout_steps = cfg.env.max_episode_steps
    eval_iter = cfg.logger.eval_iter
    frames_per_batch = cfg.collector.frames_per_batch
    update_counter = 0

    collector_iter = iter(collector)
    total_iter = len(collector)

    for _ in range(total_iter):
        timeit.printevery(num_prints=1000, total_count=total_iter, erase=True)

        with timeit("collect"):
            tensordict = next(collector_iter)

        # Update weights of the inference policy
        collector.update_policy_weights_()

        current_frames = tensordict.numel()
        pbar.update(current_frames)

        with timeit("rb - extend"):
            # Add to replay buffer
            tensordict = tensordict.reshape(-1)
            replay_buffer.extend(tensordict)

        collected_frames += current_frames

        with timeit("train"):
            # Optimization steps
            if collected_frames >= init_random_frames:
                (
                    actor_losses,
                    q_losses,
                ) = ([], [])
                for _ in range(num_updates):
                    # Update actor every delayed_updates
                    update_counter += 1
                    update_actor = update_counter % delayed_updates == 0

                    with timeit("rb - sample"):
                        sampled_tensordict = replay_buffer.sample()
                    with timeit("update"):
                        torch.compiler.cudagraph_mark_step_begin()
                        q_loss, actor_loss = update(sampled_tensordict, update_actor)

                    # Update priority
                    if prb:
                        with timeit("rb - priority"):
                            replay_buffer.update_priority(sampled_tensordict)

                    q_losses.append(q_loss.clone())
                    if update_actor:
                        actor_losses.append(actor_loss.clone())

        episode_end = (
            tensordict["next", "done"]
            if tensordict["next", "done"].any()
            else tensordict["next", "truncated"]
        )
        episode_rewards = tensordict["next", "episode_reward"][episode_end]

        # Logging
        metrics_to_log = {}
        if len(episode_rewards) > 0:
            episode_length = tensordict["next", "step_count"][episode_end]
            metrics_to_log["rollout/ep_rew_mean"] = episode_rewards.mean()
            metrics_to_log["rollout/ep_len_mean"] = episode_length.sum() / len(
                episode_length
            )

        if collected_frames >= init_random_frames:
            metrics_to_log["train/q_loss"] = torch.stack(q_losses).mean()
            if update_actor:
                metrics_to_log["train/a_loss"] = torch.stack(actor_losses).mean()

        # Evaluation
        if abs(collected_frames % eval_iter) < frames_per_batch:
            with set_exploration_type(
                ExplorationType.DETERMINISTIC
            ), torch.no_grad(), timeit("eval"):
                eval_rollout = eval_env.rollout(
                    eval_rollout_steps,
                    exploration_policy,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                eval_env.apply(dump_video)
                eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                metrics_to_log["eval/mean_reward"] = eval_reward
        if logger is not None:
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            log_metrics(logger, metrics_to_log, collected_frames)

    collector.shutdown()
    if not eval_env.is_closed:
        eval_env.close()
    if not train_env.is_closed:
        train_env.close()


if __name__ == "__main__":
    from omegaconf import OmegaConf
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from arghandler import handle

    config = handle(sys.argv)
    cfg = OmegaConf.load("config_td3.yaml")  
    
    # Manually set the parameters of the hps config file over to the cfg 
    cfg.collector.total_frames = config['steps'] # denk ik
    cfg.collector.init_random_frames = config['train']['learning_starts']
    #cfg.collector.frames_per_batch = config['train']['train_freq'] # Remains constant
    cfg.collector.env_per_collector = config['train']['num_envs']
    cfg.replay_buffer.size = config['train']['buffer_size']
    cfg.optim.gamma = config['train']['gamma']
    cfg.optim.lr = config['train']['learning_rate']
    cfg.optim.batch_size = config['train']['batch_size']
    cfg.optim.target_update_polyak = 1-config['train']['tau'] # It expects the that tau is the weight for the normal network, not the target network. So, this has to be inversed like so.
    #cfg.optim.policy_update_delay = config['train']['policy_delay'] # Remains constant
    cfg.optim.policy_noise = config['train']['target_policy_noise']
    cfg.optim.noise_clip = config['train']['target_noise_clip']
    cfg.env.name = config['env']

    cfg.lastfolder = config['lastfolder']
    cfg.rep = config['rep']
    cfg.hps = config['hps']
    cfg.env.seed = cfg.rep
    
    print(cfg)
    
    
    main(cfg)
