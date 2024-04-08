import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from parser import df


# Create a stacked bar plot
barWidth = 0.85
plt.figure(figsize=(10, 6))

# Set position of bar on Y axis
r = range(len(df))

# Make the plot
plt.barh(r, df['Love'], color='pink', edgecolor='white', height=barWidth, label='Love')
plt.barh(r, df['Heartbreak'], left=df['Love'], color='blue', edgecolor='white', height=barWidth, label='Heartbreak')
plt.barh(r, df['Optimism'], left=[i + j for i, j in zip(df['Love'], df['Heartbreak'])], color='lightblue',
        edgecolor='white', height=barWidth, label='Optimism')
plt.barh(r, df['Anger'], left=[i + j + k for i, j, k in zip(df['Love'], df['Heartbreak'], df['Optimism'])],
        color='red', edgecolor='white', height=barWidth, label='Anger')
plt.barh(r, df['Reflection'],
        left=[i + j + k + l for i, j, k, l in zip(df['Love'], df['Heartbreak'], df['Optimism'], df['Anger'])],
        color='limegreen', edgecolor='white', height=barWidth, label='Reflection')

# Add yticks on the middle of the group bars
plt.ylabel('Albums', fontweight='bold')
plt.xlabel('Sentiment Percentage', fontweight='bold')
plt.yticks([r for r in range(len(df))], df['Album Name'])
plt.title('Sentiment Percentage Distribution by Album')
plt.legend()

# Hide the percentages on the plot
plt.gca().axes.xaxis.set_visible(False)

# Show plot
plt.show()