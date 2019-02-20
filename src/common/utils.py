from baselines.run import make_vec_env, get_env_type


def make_env(env_name, nenv=1, seed=132):
    env_type, env_id = get_env_type(env_name)
    env = make_vec_env(env_id, env_type, nenv, seed, reward_scale=1)
    return env
