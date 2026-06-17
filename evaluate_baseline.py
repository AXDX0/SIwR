import gymnasium as gym

from stable_baselines3 import PPO


env = gym.make(
    "CartPole-v1"#, render_mode="human"
)

model = PPO.load(
    "models/ppo_baseline"
)

episodes = 100

scores = []

for _ in range(episodes):

    obs, _ = env.reset()

    done = False

    total_reward = 0

    while not done:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward

        done = terminated or truncated

    scores.append(total_reward)

print(
    "Average reward:",
    sum(scores) / len(scores)
)