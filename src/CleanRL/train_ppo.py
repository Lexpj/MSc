import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from arghandler import handle


config = handle(sys.argv)

if config['env_lib'] == 'discrete':
    call = f"python ppo.py "
elif config['env_lib'] == 'mujoco':
    call = f"python ppo_continuous_action.py "
elif config['env_lib'] == 'atari':
    call = f"python ppo_atari.py "

net_arch = str(config['model']['policy_kwargs']['net_arch']).replace(" ","")

call += f"--fw={config['fw']} --alg={config['alg']} --env_id={config['env']} --total_timesteps={config['steps']} --rep={config['rep']} --hps={config['hps']} --learning_rate={config['train']['learning_rate']} --num_envs={config['train']['num_envs']} --num_steps={config['train']['n_steps']} --gamma={config['train']['gamma']} --gae_lambda={config['train']['gae_lambda']} --minibatch_size={config['train']['batch_size']} --update_epochs={config['train']['n_epochs']} --clip_coef={config['train']['clip_range']} --ent_coef={config['train']['ent_coef']} --vf_coef={config['train']['vf_coef']} --max_grad_norm={config['train']['max_grad_norm']} --lastfolder={config['lastfolder']} --stats_window_size={config['train']['stats_window_size']} --net_arch={net_arch} --activation={config['model']['policy_kwargs']['activation']}"

if config['train'].get('anneal_lr',False):
    call = call + " --anneal_lr" 
if config['train'].get('normalize_advantage',False):
    call = call + " --norm_adv"
if config['train'].get('clip_range_vf',False):
    call = call + " --clip_vloss"
if config['train'].get('target_kl',False) not in [False, "none", "None", None]:
    call = call + f" --target_kl={config['train']['target_kl']}"
if config['model'].get('policy_kwargs',False):
    if config['model']['policy_kwargs'].get('ortho_init',False):
        call = call + f" --ortho_init"

os.system(call)




