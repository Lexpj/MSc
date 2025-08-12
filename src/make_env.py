import gymnasium as gym

def make_env(env_id, env_lib, gamma):

    if env_lib == 'atari':
        import ale_py
        gym.register_envs(ale_py)
        env = gym.make("ALE/"+env_id)
    else:
        env = gym.make(env_id)

    env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
    return env

