import pandas as pd
import matplotlib.pyplot as plt

df_baseline = pd.read_csv(
    "results/rewards_baseline.csv"
)
df_shire = pd.read_csv(
    "results/rewards_shire.csv"
)

plt.figure(figsize=(10.0, 6.0))

plt.plot(df_baseline["timesteps"], df_baseline["reward"], label="baseline")
plt.plot(df_shire["timesteps"], df_shire["reward"], label="SHIRE")

plt.xlabel("timesteps")
plt.ylabel("episode reward")
plt.title("baseline and SHIRE PPO")
plt.legend()
plt.grid()

plt.savefig(
    fname="results/reward_curves.svg",
    format="svg"
)

plt.show()