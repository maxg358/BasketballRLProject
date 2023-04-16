from gym.envs.registration import register

register(
    id='NBAGymEnv-v0',
    entry_point='nba_gym_env:NBAGymEnv',
)