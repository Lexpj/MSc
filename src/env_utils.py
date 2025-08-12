def identify_env(env_id):

    CLASSIC_CONTROL = [
      'acrobot',
      'cartpole',
      'mountaincar',
      'mountaincarcontinuous',
      'pendulum'
    ]
    MUJOCO = [
      'ant',
      'halfcheetah',
      'hopper',
      'humanoidstandup',
      'humanoid',
      'inverteddoublependulum',
      'invertedpendulum',
      'pusher',
      'reacher',
      'swimmer',
      'walker2d'
    ]
    ATARI = [
      'adventure',
      'airraid',
      'alien',
      'amidar',
      'assault',
      'asterix',
      'asteroids',
      'atlantis',
      'atlantis2',
      'backgammon',
      'bankheist',
      'basicmath',
      'battlezone',
      'beamrider',
      'berzerk',
      'blackjack',
      'bowling',
      'boxing',
      'breakout',
      'carnival',
      'casino',
      'centipede',
      'choppercommand',
      'crazyclimber',
      'crossbow',
      'darkchambers',
      'defender',
      'demonattack',
      'donkeykong',
      'doubledunk',
      'earthworld',
      'elevatoraction',
      'enduro',
      'entombed',
      'et',
      'fishingderby',
      'flagcapture',
      'freeway',
      'frogger',
      'frostbite',
      'galaxian',
      'gopher',
      'gravitar'
      'hangman',
      'hauntedhouse',
      'hero',
      'humancannonball',
      'icehockey',
      'jamesbond',
      'journeyescape',
      'kaboom',
      'kangaroo',
      'keysstonekapers',
      'kingkong',
      'klax',
      'koolaid',
      'krull',
      'kungfumaster',
      'lasergates',
      'lostluggage',
      'mariobros',
      'miniaturegolf',
      'montezumarevenge',
      'mrdo'
      'mspacman',
      'namethisgame',
      'othello',
      'pacman',
      'phoenix',
      'pitfall',
      'pitfall2',
      'pong',
      'pooyan',
      'privateeye',
      'qbert',
      'riverraid',
      'roadrunner'
      'robotank',
      'seaquest',
      'sirlancelot',
      'skiing'
      'solaris',
      'spaceinvaders',
      'spacewar'
      'stargunner',
      'superman',
      'surround',
      'tennis',
      'tetris',
      'tictactoe3d',
      'timepilot',
      'trondead',
      'turmoil',
      'tutankham',
      'upndown',
      'venture',
      'videocheckers',
      'videochess',
      'videocube',
      'videopinball',
      'wizardofwor',
      'wordzapper',
      'yarsrevenge',
      'zaxxon'
    ]
    
    ALL = CLASSIC_CONTROL + MUJOCO + ATARI
    
    env_id = env_id.lower()
    if '-' in env_id:
        env_id = env_id.split('-')[0]
    if 'noframeskip' in env_id:
        env_id = env_id.replace('noframeskip','')
    if '/' in env_id:
        env_id = env_id.split('/')[1]
    
    
    # Check if in one of the three categories
    if env_id not in ALL:
        return None
    
    if env_id in CLASSIC_CONTROL:
        return 'discrete'
    elif env_id in MUJOCO:
        return 'mujoco'
    elif env_id in ATARI:
        return 'atari'
    
    # Shouldnt execute beyond this:
    assert False, "Something is wrong"
    return None
    
def format_atari(env):
    if '/' in env:
        return env.split('/')[1]
    return env
 
def wrap_env(env, config, evaluate = False):
    """
    Wraps environment in a Gymnasium wrappers:
    - Normalize observation
    - Transform observation
    If not evaluate:
    - Normalize reward
    - Transform reward
    Note that this should NOT be used for SB3, since it does not comply well with logging
    """
    import gymnasium as gym
    import numpy as np

    env = gym.wrappers.NormalizeObservation(env)
    try:
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    except:
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10), env.observation_space)
    if not evaluate:
        env = gym.wrappers.NormalizeReward(env, gamma=config['train']['gamma'])
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env
 
def wrap_env_sb(env, config, evaluate = False):
    """
    Wraps environment in Gymnasium VecEnv style:
    - Normalize observation and reward are done in VecNormalize
    - Transformation observation and reward are done in custom VecEnvWrappers
    Note that normalization formula's are still done similarly.
    """
    from stable_baselines3.common.vec_env import VecEnvWrapper, VecNormalize
    import numpy as np
    
    class TransformVecObservation(VecEnvWrapper):
        def __init__(self, venv, transform_fn):
            super().__init__(venv)
            self.transform_fn = transform_fn
        def reset(self):
            obs = self.venv.reset()
            return self.transform_fn(obs)
        def step_async(self, actions):
            self.venv.step_async(actions)
        def step_wait(self):
            obs, rewards, done, infos = self.venv.step_wait()
            return self.transform_fn(obs), rewards, done, infos
    class TransformVecReward(VecEnvWrapper):
        def __init__(self, venv, transform_fn):
            super().__init__(venv)
            self.transform_fn = transform_fn
        def reset(self):
            return self.venv.reset() 
        def step_async(self, actions):
            self.venv.step_async(actions)
        def step_wait(self):
            obs, rewards, done, infos = self.venv.step_wait()
            return obs, self.transform_fn(rewards), done, infos
    
            
    if not evaluate:
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=False)
    env = TransformVecObservation(env, lambda obs: np.clip(obs, -10, 10))
    if evaluate:
        env = TransformVecReward(env, lambda reward: np.clip(reward, -10, 10))
    else:
        env = TransformVecReward(env, lambda reward: reward) # Dummy wrapper such that SB3 accounts training env and eval env as similarly wrapped environments
    return env
