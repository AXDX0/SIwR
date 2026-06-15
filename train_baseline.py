import gymnasium as gym

from stable_baselines3 import PPO

from shire.callbacks import RewardLoggerCallback


callback = RewardLoggerCallback(filename="results/rewards_baseline.csv")

env = gym.make("CartPole-v1")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)

model.learn(
    total_timesteps=50000,
    callback=callback
)

model.save(
    "models/ppo_baseline"
)