from gym.envs.registration import register

register(
    id='test',
    entry_point='gym_gc.envs:TestEnv',
)
