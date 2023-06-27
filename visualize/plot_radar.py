# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi

# Set data
df = pd.DataFrame({
    'group': ['layer1', 'layer2', 'layer3', 'layer4', 'layer5', 'layer6'],
    '$f_a$': [91.5, 91.7, 91.8, 91.7, 91.8, 91.7],
    '$\hat{f}_c$': [86.1, 86.2, 86.1, 86.4, 86.2, 86.4],
    "$f_{(c,0)}'$": [86.7, 86.9, 87.1, 87.1, 87.2, 87.1],
    "$f_{(c,L)}'$": [88.2, 88.3, 88.7, 88.6, 88.4, 88.5],
    '$\hat{f}_t$': [88.7, 88.7, 88.9, 88.7, 88.7, 88.8],
    "$f_{(t,0)}'$": [89.4, 89.5, 89.7, 89.7, 89.5, 89.6],
    "$f_{(t,L)}'$": [90.5, 90.9, 90.8, 90.5, 90.4, 90.3]
})

# ------- PART 1: Create background

# number of variable
categories = list(df)[1:]
N = len(categories)
name = list(df['group'])
# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
ax = plt.subplot(111, polar=True)

# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw one axe per variable + add labels
plt.xticks(angles[:-1], categories)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks([10, 20, 30, 40, 50, 60, 70,80, 90, 100, 110, 120, 130, 140], ["86", "87", "88", "89", "90", "91", "92"], color="grey", size=4)
plt.ylim(0, 100)

# ------- PART 2: Add plots

# Plot each individual = each line of the data
# I don't make a loop, because plotting more than 3 groups makes the chart unreadable

# Ind1
values = df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label=name[0])
ax.fill(angles, values, 'blue', alpha=0.1)

# Ind2
values = df.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label=name[1])
ax.fill(angles, values, 'green', alpha=0.1)

# # Ind1
# values = df.loc[2].drop('group').values.flatten().tolist()
# values += values[:1]
# ax.plot(angles, values, linewidth=1, linestyle='solid', label=name[2])
# ax.fill(angles, values, 'purple', alpha=0.1)
#
# # Ind2
# values = df.loc[3].drop('group').values.flatten().tolist()
# values += values[:1]
# ax.plot(angles, values, linewidth=1, linestyle='solid', label=name[3])
# ax.fill(angles, values, 'r', alpha=0.1)
# # Ind1
# values = df.loc[4].drop('group').values.flatten().tolist()
# values += values[:1]
# ax.plot(angles, values, linewidth=1, linestyle='solid', label=name[4])
# ax.fill(angles, values, 'b', alpha=0.1)
#
# # Ind2
# values = df.loc[5].drop('group').values.flatten().tolist()
# values += values[:1]
# ax.plot(angles, values, linewidth=1, linestyle='solid', label=name[5])
# ax.fill(angles, values, 'r', alpha=0.1)
# # Ind2
# values = df.loc[6].drop('group').values.flatten().tolist()
# values += values[:1]
# ax.plot(angles, values, linewidth=1, linestyle='solid', label=name[5])
# ax.fill(angles, values, 'r', alpha=0.1)
# Add legend
plt.legend(loc='upper right', bbox_to_anchor=(0., 1.1))

# Show the graph
plt.show()
