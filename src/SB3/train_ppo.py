import gymnasium as gym
from stable_baselines3 import PPO
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from arghandler import handle
from env_utils import wrap_env_sb
import torch
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

def main(config):


    def make_env():
        def thunk():
            _env = gym.make(config['env'])
            _env = gym.wrappers.RecordEpisodeStatistics(_env)
            return _env
        return thunk
        
    env = DummyVecEnv([make_env()])
    eval_env = DummyVecEnv([make_env()])
    
    if config['environment'].get('wrapped',False):
        env = wrap_env_sb(env, config)
        eval_env = wrap_env_sb(eval_env, config, evaluate=True)
    
    # Network initialization is the same as CleanRL    
    #if hasattr(torch.nn, config['model']['policy_kwargs']['activation']):
    #    activation_fn = getattr(torch.nn, config['model']['policy_kwargs']['activation'])
    
    
    def linear_schedule(initial_value: float):
        def func(progress_remaining: float):
            return progress_remaining * initial_value
        return func
    
    if config['train']['anneal_lr']:
        learning_rate_schedule = linear_schedule(config['train']['learning_rate']) #anneal linearly
    else:
        learning_rate_schedule = config['train']['learning_rate'] # constant
    
    # Is the same as CleanRL by default
    #policy_kwargs = dict(activation_fn=activation_fn,
    #                     net_arch=dict(pi=config['model']['policy_kwargs']['net_arch'], 
    #                                   vf=config['model']['policy_kwargs']['net_arch']),
    #                     ortho_init=config['model']['policy_kwargs']['ortho_init'])
    
    logpath = f"./results/{config['alg']}_{config['env']}_{config['steps']}/{config['hps']}" if config['log'] else None
    
    if config['train']['clip_range_vf'] == True: # CleanRL, use clip_range = clip_range_vf
        clip_range_vf = config['train']['clip_range']
    elif config['train']['clip_range_vf'] in [None,False]: # SB3: dont use clip_range_vf
        clip_range_vf = None 
    else: # SB3: use clip_range_vf float value
        clip_range_vf = config['train']['clip_range']
    
    print(clip_range_vf)
    print(config)
    with open("TESTFILE.txt","w+") as f:
        f.write(str(config))
    
    
    if config['env_lib'] == 'discrete':
        eval_freq = 5000
    elif config['env_lib'] == 'mujoco':
        eval_freq = 10000
    
    # This does not account for not-provided variables. Still have to adjust that!!
    model = PPO(policy=config['model']['policy'],
                env=env,
                learning_rate=learning_rate_schedule,
                n_steps=config['train']['n_steps'],
                batch_size=config['train']['batch_size'],
                n_epochs=config['train']['n_epochs'],
                gamma=config['train']['gamma'],
                gae_lambda=config['train']['gae_lambda'],
                clip_range=config['train']['clip_range'],
                clip_range_vf=clip_range_vf,
                #normalize_advantage=config['train']['normalize_advantage'],
                ent_coef=config['train']['ent_coef'],
                vf_coef=config['train']['vf_coef'],
                #max_grad_norm=config['train']['max_grad_norm'],
                target_kl=config['train']['target_kl'],
                stats_window_size=config['train']['stats_window_size'],
                  
                tensorboard_log=logpath
            )
    print(model.policy.share_features_extractor)
                      
    eval_callback = EvalCallback(eval_env, best_model_save_path=logpath,
                              log_path=logpath, eval_freq=eval_freq,
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
     
