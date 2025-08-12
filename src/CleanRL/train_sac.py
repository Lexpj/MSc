import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from arghandler import handle


config = handle(sys.argv)

call = f"python sac_continuous_action.py "


net_arch = str(config['model']['policy_kwargs']['net_arch']).replace(" ","")

call += f"--fw={config['fw']}\
 --alg={config['alg']}\
 --env_id={config['env']}\
 --total_timesteps={config['steps']}\
 --rep={config['rep']}\
 --hps={config['hps']}\
 --policy_lr={config['train']['learning_rate']}\
 --num_envs={config['train']['num_envs']}\
 --buffer_size={config['train']['buffer_size']}\
 --gamma={config['train']['gamma']}\
 --tau={config['train']['tau']}\
 --batch_size={config['train']['batch_size']}\
 --learning_starts={config['train']['learning_starts']}\
 --lastfolder={config['lastfolder']}\
 --stats_window_size={config['train']['stats_window_size']}"

if config['train']['ent_coef'] == 'auto':
    call = call + " --autotune" 	# Value of alpha is not considered when this bool is flagged
else:
    call = call + " --alpha={config['train']['ent_coef']}" # If not auto, by default autotune is False and ent_coef is expected to be a float


if config['environment'].get('wrapped',False): # Not considered, but could be used in the future
    call = call + f" --wrapped"

print(call)

os.system(call)




