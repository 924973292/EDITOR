import matplotlib.pyplot as plt

mAP = [69.1, 69.3, 69.5, 70, 69.3]  # mAP of action
R1 = [86.5, 86.7, 86.7, 86.8, 86.6]  # js diversity
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
width = 0.4  # 柱形的宽度
x1_list = []
x2_list = []
for i in range(len(mAP)):
    x1_list.append(i)
    x2_list.append(i + width)

fig, ax1 = plt.subplots()
# 设置左侧Y轴对应的figure
ax1.set_ylabel('mAP (%)')
ax1.set_ylim(68.0, 71)
bar1 = ax1.bar(x1_list, mAP, width=width, color='#1f77b4', align='edge', tick_label=labels)
ax1.bar_label(bar1, label_type='edge',fontsize=10,fmt="%.1f",padding=1,fontweight='bold')
ax1.set_xlabel("$D$")
ax1.set_xticklabels(ax1.get_xticklabels())  # 设置共用的x轴
# plt.title("Performance of different fusion dimensions on MSMT17",fontsize=14)
# 设置右侧Y轴对应的figure
ax2 = ax1.twinx()
ax2.set_ylabel('Rank-1 (%)')
ax2.set_ylim(86.0, 87.0)
bar2 = ax2.bar(x2_list, R1, width=width, color='orange', align='edge', tick_label=labels)
ax2.bar_label(bar2, label_type='edge',fontsize=10,fmt="%.1f",padding=1,fontweight='bold')
plt.tight_layout()


fig.legend(['mAP', 'Rank-1'], loc=(0.18,0.804))
plt.savefig("test.pdf")
plt.show()

import matplotlib.pyplot as plt

mAP = [120.9,134.7,153.8,208,283.4]  # mAP of action
R1 = [23.7,25.5,28.1,35.3,45.3]  # js diversity
labels = [384, 576, 768, 1152, 1536]
plt.figure(dpi=200)
plt.rcParams['axes.labelsize'] = 14  # xy轴label的size
plt.rcParams['xtick.labelsize'] = 13  # x轴ticks的size
plt.rcParams['ytick.labelsize'] = 13  # y轴ticks的size
# plt.rcParams['legend.fontsize'] = 12  # 图例的size
def label(ax):
    ax.set_xlabel('X', fontsize=16)
    ax.set_ylabel('Y', fontsize=16)
    ax.legend(loc='best')
# 设置柱形的间隔
width = 0.4  # 柱形的宽度
x1_list = []
x2_list = []
for i in range(len(mAP)):
    x1_list.append(i)
    x2_list.append(i + width)

fig, ax1 = plt.subplots()
# 设置左侧Y轴对应的figure
ax1.set_ylabel('Params (M)')
ax1.set_ylim(40, 310)
bar1 = ax1.bar(x1_list, mAP, width=width, color='#1f77b4', align='edge', tick_label=labels)
ax1.bar_label(bar1, label_type='edge',fontsize=10,fmt="%.1f",padding=1,fontweight='bold')
ax1.set_xlabel("$D$")
ax1.set_xticklabels(ax1.get_xticklabels())  # 设置共用的x轴
# plt.title("Performance of different fusion dimensions on MSMT17",fontsize=14)
# 设置右侧Y轴对应的figure
ax2 = ax1.twinx()
ax2.set_ylabel('FLOPs (G)')
ax2.set_ylim(10, 60)
bar2 = ax2.bar(x2_list, R1, width=width, color='orange', align='edge', tick_label=labels)
ax2.bar_label(bar2, label_type='edge',fontsize=10,fmt="%.1f",padding=1,fontweight='bold')
plt.tight_layout()


fig.legend(['Params', 'FLOPs'], loc=(0.18,0.804))
plt.savefig("test_para.pdf")
plt.show()

