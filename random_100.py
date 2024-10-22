import pandas as pd

df = pd.read_csv('evaluation_results.csv') 

# Randomly sample 100 rows
sampled_df = df.sample(n=100, random_state=42)

# Save the sampled data to a new CSV file
sampled_df.to_csv('qualitative_100.csv', index=False)
