from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("CartPole-v1")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)

model.learn(
    total_timesteps=50000
)

model.save("ppo_baseline")

#
import random

class IntuitionNet:

    def predict_action(self, angle):

        if angle < 0:

            return random.choices(
                [0,1],
                weights=[0.8,0.2]
            )[0]

        return random.choices(
            [0,1],
            weights=[0.2,0.8]
        )[0]

def intuition_loss(
        policy_action,
        intuitive_action):

    return int(
        policy_action != intuitive_action
    )

reward = reward - lambda_ * intuition_loss

modified_reward = reward - 0.1 * loss

plt.plot(...)
