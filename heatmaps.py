import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from parser import df
import seaborn as sns
import matplotlib.pyplot as plt

# Define the number of columns and rows for subplots
num_cols = 2
num_rows = (len(df['Album Name'].unique()) + num_cols - 1) // num_cols

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))

# Iterate through each unique album name and plot the heatmap
for i, album_name in enumerate(df['Album Name'].unique()):
    row = i // num_cols
    col = i % num_cols
    album_df = df[df['Album Name'] == album_name]
    emotions_df = album_df[['Love', 'Heartbreak', 'Optimism', 'Anger', 'Reflection']]
    ax = sns.heatmap(emotions_df, cmap='coolwarm', annot=True, fmt=".2f", ax=axes[row, col])
    ax.set_title(album_name)
    ax.set_xlabel('Emotions')
    ax.set_ylabel('Emotions')

# Adjust layout
plt.tight_layout()
plt.show()
