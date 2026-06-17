import gymnasium as gym
import time

from stable_baselines3 import PPO

from shire.callbacks import RewardLoggerCallback


callback = RewardLoggerCallback(filename="results/rewards_baseline.csv")

env = gym.make("CartPole-v1"#,
    # render_mode="human"
)

model = PPO(
    "MlpPolicy",
    env,
    tensorboard_log="./tb_logs/",
    verbose=1
)

start_time = time.time()

model.learn(
    total_timesteps=50000,
    callback=callback
)

elapsed = time.time() - start_time
print(f"Training time: {elapsed:.2f}s")

model.save(
    "models/ppo_baseline"
)