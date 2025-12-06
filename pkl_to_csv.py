import pandas as pd

df = pd.read_pickle("data/routerbench_0shot.pkl")
df.to_csv("data/routerbench_0shot.csv", index=False)
