import gymnasium as gym
from stable_baselines3 import SAC
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from arghandler import handle
from env_utils import wrap_env
import torch
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
from stable_baselines3.common.env_util import make_vec_env

def main(config):

    #env = make_env(config['env'], config['env_lib'], config['train']['gamma'])
    
    
    env = gym.make(config['env'])
    eval_env = gym.make(config['env'])
    env = make_vec_env(config['env'], n_envs=config['train']['num_envs'])

    if config['environment'].get('wrapped',False): # Always false, no wrappers used
        env = wrap_env(env, config)
        eval_env = wrap_env(eval_env, config, evaluate=True)
    
        
    if hasattr(torch.nn, config['model']['policy_kwargs']['activation']):
        activation_fn = getattr(torch.nn, config['model']['policy_kwargs']['activation'])
    
    
    def linear_schedule(initial_value: float): # Not used, since both don't use linear annealing
        def func(progress_remaining: float):
            return progress_remaining * initial_value
        return func
    
    if config['train'].get('anneal_lr',False): # Not used
        learning_rate_schedule = linear_schedule(config['train']['learning_rate']) #anneal linearly
    else:
        learning_rate_schedule = config['train']['learning_rate'] # constant
    
    policy_kwargs = dict(activation_fn=activation_fn,
                         net_arch=config['model']['policy_kwargs']['net_arch']
                        )
    
    logpath = f"./results/{config['alg']}_{config['env']}_{config['steps']}/{config['hps']}" if config['log'] else None
    
    print(config)
    
    model = SAC(policy=config['model']['policy'],
                env=env,
                learning_rate=learning_rate_schedule,
                buffer_size=config['train']['buffer_size'],
                learning_starts=config['train']['learning_starts'],
                batch_size=config['train']['batch_size'],
                tau=config['train']['tau'],
                gamma=config['train']['gamma'],
                ent_coef=config['train']['ent_coef'],
                stats_window_size=config['train']['stats_window_size'],
                  
                tensorboard_log=logpath
            )
                      
    eval_callback = EvalCallback(eval_env, best_model_save_path=logpath,
                              log_path=logpath, eval_freq=10000//config['train']['num_envs'],
                              n_eval_episodes=30, deterministic=True,
                              render=False)

    model.learn(total_timesteps=config['steps'],
                tb_log_name=f"{config['rep']}",
                log_interval=config['logging']['log_interval'],
                callback=eval_callback
            )
            
    if config['save_model']:
        model.save(f"./results/{config['alg']}_{config['env']}_{config['steps']}/{config['hps']}/{config['rep']}_{config['lastfolder']}/model")


if __name__ == "__main__":
     
     

     config = handle(sys.argv)
     main(config)
     
