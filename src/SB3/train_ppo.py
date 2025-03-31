import gymnasium as gym
from stable_baselines3 import PPO
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from arghandler import handle
import torch

def main(config):

    if config['env_lib'] == 'atari':
        import ale_py
        gym.register_envs(ale_py)
        env = gym.make("ALE/"+config['env'])
    else:
        env = gym.make(config['env'])
    
    if hasattr(torch.nn, config['model']['policy_kwargs']['activation']):
        activation_fn = getattr(torch.nn, config['model']['policy_kwargs']['activation'])
    
    policy_kwargs = dict(activation_fn=activation_fn,
                         net_arch=dict(pi=config['model']['policy_kwargs']['net_arch'], 
                                       vf=config['model']['policy_kwargs']['net_arch']),
                         ortho_init=config['model']['policy_kwargs']['ortho_init'])
    
    logpath = f"./results/{config['alg']}_{config['env']}_{config['steps']}/{config['hps']}" if config['log'] else None
    
    # This does not account for not-provided variables. Still have to adjust that!!
    model = PPO(policy=config['model']['policy'],
                env=env,
                learning_rate=config['train']['learning_rate'],
                n_steps=config['train']['n_steps'],
                batch_size=config['train']['batch_size'],
                n_epochs=config['train']['n_epochs'],
                gamma=config['train']['gamma'],
                gae_lambda=config['train']['gae_lambda'],
                clip_range=config['train']['clip_range'],
                clip_range_vf=config['train']['clip_range_vf'],
                normalize_advantage=config['train']['normalize_advantage'],
                ent_coef=config['train']['ent_coef'],
                vf_coef=config['train']['vf_coef'],
                max_grad_norm=config['train']['max_grad_norm'],
                use_sde=config['train']['use_sde'],
                sde_sample_freq=config['train']['sde_sample_freq'],
                rollout_buffer_class=config['train']['rollout_buffer_class'],
                rollout_buffer_kwargs=config['train']['rollout_buffer_kwargs'],
                target_kl=config['train']['target_kl'],
                stats_window_size=config['train']['stats_window_size'],
                  
                policy_kwargs=policy_kwargs,
                  
                tensorboard_log=logpath
            )
    
    model.learn(total_timesteps=config['steps'],
                tb_log_name=f"{config['rep']}",
                log_interval=config['logging']['log_interval']
            )
            
    if config['save_model']:
        model.save(f"./results/{config['alg']}_{config['env']}_{config['steps']}/{config['hps']}/{config['rep']}_{config['lastfolder']}/model")


if __name__ == "__main__":
     
     

     config = handle(sys.argv)
     main(config)
     
