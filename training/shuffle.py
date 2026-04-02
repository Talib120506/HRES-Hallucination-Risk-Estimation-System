import pandas as pd

df = pd.read_excel("../data/processed/features_correct_incorrect.xlsx")
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
df_shuffled.to_excel("../data/processed/features_shuffled_final.xlsx", index=False)
print(f"Shuffled {len(df_shuffled)} rows -> ../data/processed/features_shuffled_final.xlsx")
