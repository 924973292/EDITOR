# libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make a data frame
df = pd.DataFrame({'x': range(1, 11), 'y1': np.random.randn(10), 'y2': np.random.randn(10) + range(1, 11),
                   'y3': np.random.randn(10) + range(11, 21), 'y4': np.random.randn(10) + range(6, 16),
                   'y5': np.random.randn(10) + range(4, 14) + (0, 0, 0, 0, 0, 0, 0, -3, -8, -6),
                   'y6': np.random.randn(10) + range(2, 12), 'y7': np.random.randn(10) + range(5, 15),
                   'y8': np.random.randn(10) + range(4, 14), 'y9': np.random.randn(10) + range(4, 14)})

# Initialize the figure style
plt.style.use('seaborn-darkgrid')

# create a color palette
palette = plt.get_cmap('Set1')

# multiple line plot
num = 0
for column in df.drop('x', axis=1):
    num += 1

    # Find the right spot on the plot
    plt.subplot(3, 3, num)

    # Plot the lineplot
    plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=1.9, alpha=0.9, label=column)

    # Same limits for every chart
    plt.xlim(0, 10)
    plt.ylim(-2, 22)

    # Not ticks everywhere
    if num in range(7):
        plt.tick_params(labelbottom='off')
    if num not in [1, 4, 7]:
        plt.tick_params(labelleft='off')

    # Add title
    plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num))

# general title
plt.suptitle("How the 9 students improved\nthese past few days?", fontsize=13, fontweight=0, color='black',
             style='italic', y=1.02)

# Axis titles
plt.text(0.5, 0.02, 'Time', ha='center', va='center')
plt.text(0.06, 0.5, 'Note', ha='center', va='center', rotation='vertical')

# Show the graph
plt.show()