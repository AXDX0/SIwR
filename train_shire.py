import gymnasium as gym

from shire.shire_ppo import SHIREPPO

from shire.callbacks import RewardLoggerCallback


callback = RewardLoggerCallback(filename="results/rewards_shire.csv")

env = gym.make(
    "CartPole-v1"
)

model = SHIREPPO(
    "MlpPolicy",
    env,
    verbose=1,
    shire_coef=0.5
)

model.learn(
    total_timesteps=50000,
    callback=callback
)

model.save(
    "models/ppo_shire"
)