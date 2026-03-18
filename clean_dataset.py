import pandas as pd

df = pd.read_csv("data/crop_dataset_realistic.csv")

# Remove duplicates
df = df.drop_duplicates()

# Remove impossible values
df = df[
    (df["temperature"] > 0) &
    (df["humidity"] >= 0) &
    (df["soil_ph"].between(4, 9))
]

# Clip extreme values
df["humidity"] = df["humidity"].clip(0, 100)

# Save clean dataset
df.to_csv("data/crop_dataset_cleaned.csv", index=False)

print("Cleaned dataset size:", len(df))