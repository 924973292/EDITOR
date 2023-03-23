import matplotlib.pyplot as plt

mAP = [67.0, 67.6, 68.2, 68.2, 67.8]  # mAP of action
R1 = [85.4, 85.8, 86.6, 86.0, 85.6]  # js diversity
labels = [384, 576, 768, 1152, 1536]
plt.figure(dpi=400)
plt.rcParams['axes.labelsize'] = 14  # xy轴label的size
plt.rcParams['xtick.labelsize'] = 13  # x轴ticks的size
plt.rcParams['ytick.labelsize'] = 13  # y轴ticks的size
# plt.rcParams['legend.fontsize'] = 12  # 图例的size
def label(ax):
    ax.set_xlabel('X', fontsize=16)
    ax.set_ylabel('Y', fontsize=16)
    ax.legend(loc='best')
# 设置柱形的间隔
width = 0.33  # 柱形的宽度
x1_list = []
x2_list = []
for i in range(len(mAP)):
    x1_list.append(i)
    x2_list.append(i + width)

fig, ax1 = plt.subplots()
# 设置左侧Y轴对应的figure
ax1.set_ylabel('mAP (%)')
ax1.set_ylim(66.4, 68.7)
bar1 = ax1.bar(x1_list, mAP, width=width, color='#1f77b4', align='edge', tick_label=labels)
ax1.bar_label(bar1, label_type='edge',fontsize=11.5,fmt="%.1f",padding=1)
ax1.set_xlabel("$D$")
ax1.set_xticklabels(ax1.get_xticklabels())  # 设置共用的x轴
# plt.title("Performance of different fusion dimensions on MSMT17",fontsize=14)
# 设置右侧Y轴对应的figure
ax2 = ax1.twinx()
ax2.set_ylabel('Rank-1 (%)')
ax2.set_ylim(85.2, 86.8)
bar2 = ax2.bar(x2_list, R1, width=width, color='orange', align='edge', tick_label=labels)
ax2.bar_label(bar2, label_type='edge',fontsize=11.5,fmt="%.1f",padding=1)
plt.tight_layout()

fig.legend(['mAP', 'Rank-1'], loc=(0.16,0.804))
plt.savefig("test.pdf")
plt.show()
