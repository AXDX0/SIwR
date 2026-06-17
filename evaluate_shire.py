import gymnasium as gym

from shire.shire_ppo import SHIREPPO


env = gym.make(
    "CartPole-v1"#, render_mode="human"
)

model = SHIREPPO.load(
    "models/ppo_shire"
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