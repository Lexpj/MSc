import gymnasium as gym
from stable_baselines3 import TD3
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from arghandler import handle
from env_utils import wrap_env
import torch
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise

def main(config):

    #env = make_env(config['env'], config['env_lib'], config['train']['gamma'])
    
    
    env = gym.make(config['env'])
    a_low = env.action_space.low
    a_high = env.action_space.high
    n_actions = env.action_space.shape[0]
   
    eval_env = gym.make(config['env'])
    env = make_vec_env(config['env'], n_envs=config['train']['num_envs'])

    if config['environment'].get('wrapped',False): # Always false, no wrappers used
        env = wrap_env(env, config)
        eval_env = wrap_env(eval_env, config, evaluate=True)
    
        
    #if hasattr(torch.nn, config['model']['policy_kwargs']['activation']):
    #    activation_fn = getattr(torch.nn, config['model']['policy_kwargs']['activation'])
    
    learning_rate_schedule = config['train']['learning_rate'] # constant
    
    policy_kwargs = dict(
                         net_arch=config['model']['policy_kwargs']['net_arch']
                        )
    
    #if config['train']['action_noise'] == 0:
    action_noise = None
    #else:
    #    action_scale = (a_high-a_low)/2 # As is from CleanRL
    #    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=config['train']['action_noise'] * action_scale)
    
    logpath = f"./results/{config['alg']}_{config['env']}_{config['steps']}/{config['hps']}" if config['log'] else None
    
    print(config)
    
    model = TD3(policy=config['model']['policy'],
                env=env,
                learning_rate=learning_rate_schedule,
                buffer_size=config['train']['buffer_size'],
                learning_starts=config['train']['learning_starts'],
                batch_size=config['train']['batch_size'],
                tau=config['train']['tau'],
                gamma=config['train']['gamma'],
                target_policy_noise=config['train']['target_policy_noise'],
                target_noise_clip=config['train']['target_noise_clip'],
                action_noise=action_noise,
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
     
