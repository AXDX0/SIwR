import pandas as pd
import matplotlib.pyplot as plt

df_baseline = pd.read_csv(
    "results/rewards_baseline.csv"
)
df_shire = pd.read_csv(
    "results/rewards_shire.csv"
)

plt.figure(figsize=(10.0, 6.0))

plt.plot(df_baseline["timesteps"], df_baseline["reward"])
plt.plot(df_shire["timesteps"], df_shire["reward"])

plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("baseline and SHIRE PPO")
plt.legend(["baseline", "SHIRE"])

plt.savefig(
    fname="results/reward_curves.svg",
    format="svg"
)