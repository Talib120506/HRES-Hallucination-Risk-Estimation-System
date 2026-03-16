import pandas as pd

df = pd.read_excel("features_correct_incorrect.xlsx")
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
df_shuffled.to_excel("features_shuffled_final.xlsx", index=False)
print(f"Shuffled {len(df_shuffled)} rows -> features_shuffled_final.xlsx")
