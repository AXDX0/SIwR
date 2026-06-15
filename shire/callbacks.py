from stable_baselines3.common.callbacks import BaseCallback
import csv
import os


class RewardLoggerCallback(BaseCallback):

    def __init__(self, filename="results/rewards.csv"):
        super().__init__()
        self.rewards = []

        self.filename = filename

    def _on_training_start(self):

        os.makedirs(
            "results",
            exist_ok=True
        )

        with open(
            self.filename,
            "w",
            newline=""
        ) as f:

            writer = csv.writer(f)

            writer.writerow(
                ["timesteps", "reward"]
            )

    def _on_step(self):

        if len(
            self.model.ep_info_buffer
        ) > 0:

            reward = (
                self.model
                .ep_info_buffer[-1]["r"]
            )

            with open(
                self.filename,
                "a",
                newline=""
            ) as f:

                writer = csv.writer(f)

                writer.writerow([
                    self.num_timesteps,
                    reward
                ])

        return True