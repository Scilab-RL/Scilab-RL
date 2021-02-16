import gym

def make_robustGoalConditionedHierarchicalEnv(env):
    if hasattr(env, 'env'):
        env = env.env
    env.step_old = env.step

    def step_new(action):
        obs, reward, done, info = env.step_old(action)
        reward = float(reward)
        return obs, reward, done, info

    env.step = step_new

    return env

def make_robustGoalConditionedModel(model):

    return model