#%%
import matplotlib.pyplot as plt
import numpy as np

# Data for the bars
categories = ['', ' ']
green_heights = [81.65*8, 81.65*8]  # Heights of the green segments
blue_heights = [427.2, 30.7]    # Heights of the blue segments

# Calculate the total heights for each category
total_heights = np.array(green_heights) + np.array(blue_heights)

# Create the bar plot
fig, ax = plt.subplots(figsize=(6, 4))

# Plot stacked bars
bars1 = ax.bar(categories, green_heights, color='#706caa', edgecolor='black', label='                          ')
bars2 = ax.bar(categories, blue_heights, bottom=green_heights, color='#ffec47', edgecolor='black', label='                    ')



# Add a title and labels
plt.ylabel('', fontsize=12)

# Customize the grid
plt.grid(axis='y', linestyle='--', color='lightgray', linewidth=0.5)

# Show the plot
plt.ylim(0, 1200)  # Adjust the y-axis limits
plt.legend()
plt.show()
# %%
