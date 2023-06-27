# # from matplotlib import pyplot as plt
# #
# #
# # def get_data(str):
# #     str = str.replace('\n', '').split('\t')
# #     data = []
# #     for item in str[1:]:
# #         data.append(float(item))
# #     return data
# #
# #
# # def get_all_data(txt):
# #     data = []
# #     for item in txt:
# #         data.append(get_data(item))
# #     return data
# #
# #
# # def txt_process(filename_s, choice=1):
# #     f = open(filename_s)
# #     line = f.readlines()  # 调用文件的 readline()方法
# #     if choice == 0:
# #         txt = line[1:8]
# #     else:
# #         txt = line[9:]
# #     data = get_all_data(txt)
# #     f.close()
# #     return data
# #
# #
# # def depth_effect(filename_s, pattern=0):
# #     dataset = filename_s.split('_')[0]
# #     map = txt_process(filename_s, choice=0)
# #     R1 = txt_process(filename_s, choice=1)
# #     depth = range(1, 7)
# #     plt.figure(dpi=350)
# #     dict = {0: "mAP", 1: "Rank-1"}
# #     # figure_name = "Depth's effect on " + dataset
# #     if pattern == 0:
# #         plt.plot(depth, map[0], label='$\hat{f_c}$')
# #         plt.plot(depth, map[1], label='$\hat{f_t}$')
# #         plt.plot(depth, map[2], label="$f_{(c,0)}'$")
# #         plt.plot(depth, map[3], label="$f_{(t,0)}'$")
# #         plt.plot(depth, map[4], label="$f_{(c,L)}'$")
# #         plt.plot(depth, map[5], label="$f_{(t,L)}'$")
# #         plt.plot(depth, map[6], label='$f_{a}$')
# #         # plt.plot(depth, map[0])
# #         # plt.plot(depth, map[1])
# #         # plt.plot(depth, map[2])
# #         # plt.plot(depth, map[3])
# #         # plt.plot(depth, map[4])
# #         # plt.plot(depth, map[5])
# #         # plt.plot(depth, map[6])
# #         plt.ylabel("mAP (%)",fontsize=18)
# #         for x, y in zip(depth, map[-1]):
# #             plt.text(x - 0.25, y - 0.27, '%.1f' % y, fontdict={'fontsize': 12})
# #
# #     else:
# #         plt.plot(depth, R1[0], label='$\hat{f_c}$')
# #         plt.plot(depth, R1[1], label='$\hat{f_t}$')
# #         plt.plot(depth, R1[2], label="$f_{(c,0)}'$")
# #         plt.plot(depth, R1[3], label="$f_{(t,0)}'$")
# #         plt.plot(depth, R1[4], label="$f_{(c,L)}'$")
# #         plt.plot(depth, R1[5], label="$f_{(t,L)}'$")
# #         plt.plot(depth, R1[6], label='$f_{a}$')
# #         # plt.plot(depth, R1[0])
# #         # plt.plot(depth, R1[1])
# #         # plt.plot(depth, R1[2])
# #         # plt.plot(depth, R1[3])
# #         # plt.plot(depth, R1[4])
# #         # plt.plot(depth, R1[5])
# #         # plt.plot(depth, R1[6])
# #         plt.ylabel("Rank-1 (%)",fontsize=18)
# #         for x, y in zip(depth, R1[-1]):
# #             plt.text(x - 0.25, y -0.05, '%.1f' % y, fontdict={'fontsize': 12})
# #
# #     _xtick = list(depth)
# #     plt.xticks(_xtick[::1], rotation=0)
# #     plt.xlabel("depth",fontsize=18)
# #     # plt.title(figure_name,fontsize=18)
# #     plt.legend(loc=2, bbox_to_anchor=(1.01, 1.0), borderaxespad=0.,fontsize = 18)
# #
# #     plt.subplots_adjust(right=0.85)
# #     plt.tight_layout()
# #     plt.savefig(dataset + dict[pattern] + '.pdf',bbox_inches = 'tight')
# #     plt.show()
# #
# #
# # filename_s = 'Market1501_mAP_R1.txt'
# # depth_effect(filename_s, pattern=0)
# from matplotlib import pyplot as plt
#
#
# # def get_data(str):
# #     str = str.replace('\n', '').split('\t')
# #     data = []
# #     for item in str[1:]:
# #         data.append(float(item))
# #     return data
# #
# #
# # def get_all_data(txt):
# #     data = []
# #     for item in txt:
# #         data.append(get_data(item))
# #     return data
# #
# #
# # def txt_process(filename_s, choice=1):
# #     with open(filename_s) as f:
# #         lines = f.readlines()
# #     if choice == 0:
# #         txt = lines[1:8]
# #     else:
# #         txt = lines[9:]
# #     data = get_all_data(txt)
# #     return data
# #
# #
# # def depth_effect(filename_s, pattern=0):
# #     dataset = filename_s.split('_')[0]
# #     map_data = txt_process(filename_s, choice=0)
# #     r1_data = txt_process(filename_s, choice=1)
# #     depth = range(1, 7)
# #
# #     plt.figure(figsize=(6, 6), dpi=400)
# #     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
# #     labels = ['$\hat{f_c}$', "$f_{(c,0)}'$", "$f_{(c,L)}'$", '$\hat{f_t}$', "$f_{(t,0)}'$", "$f_{(t,L)}'$", '$f_{a}$']
# #
# #     for i in range(len(labels)):
# #         if pattern == 0:
# #             plt.plot(depth, map_data[i], label=labels[i], color=colors[i], linestyle='-', linewidth=2)
# #             plt.ylabel("mAP (%)", fontsize=18)
# #             for x, y in zip(depth, map_data[-1]):
# #                 plt.text(x - 0.25, y - 0.27, '%.1f' % y, fontdict={'fontsize': 14})
# #         else:
# #             plt.plot(depth, r1_data[i], label=labels[i], color=colors[i], linestyle='-', linewidth=2)
# #             plt.ylabel("Rank-1 (%)", fontsize=18)
# #             for x, y in zip(depth, r1_data[-1]):
# #                 plt.text(x - 0.25, y - 0.05, '%.1f' % y, fontdict={'fontsize': 14})
# #
# #     plt.xlabel("depth", fontsize=18)
# #     plt.xticks(list(depth), fontsize=10)
# #     plt.yticks(fontsize=10)
# #     plt.legend(loc='upper right', fontsize=10)
# #     plt.grid(False)
# #     plt.tight_layout()
# #     # plt.savefig(dataset + dict[pattern] + '.pdf',bbox_inches = 'tight')
# #     plt.show()
# #
# #
# # filename_s = 'Market1501_mAP_R1.txt'
# # depth_effect(filename_s, pattern=0)
#
# from matplotlib import pyplot as plt
#
# def get_data(str):
#     str = str.replace('\n', '').split('\t')
#     data = []
#     for item in str[1:]:
#         data.append(float(item))
#     return data
#
# def get_all_data(txt):
#     data = []
#     for item in txt:
#         data.append(get_data(item))
#     return data
#
# def txt_process(filename_s, choice=1):
#     with open(filename_s) as f:
#         lines = f.readlines()
#     if choice == 0:
#         txt = lines[1:8]
#     else:
#         txt = lines[9:]
#     data = get_all_data(txt)
#     return data
#
# def depth_effect(filename_s, pattern=0):
#     dataset = filename_s.split('_')[0]
#     map_data = txt_process(filename_s, choice=0)
#     r1_data = txt_process(filename_s, choice=1)
#     depth = range(1, 7)
#     dict = {0: "mAP", 1: "Rank-1"}
#     fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(12, 6), dpi=400, sharey=False)
#
#     colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
#     labels = ['$\hat{f_c}$', "$f_{(c,0)}'$", "$f_{(c,L)}'$", '$\hat{f_t}$', "$f_{(t,0)}'$", "$f_{(t,L)}'$", '$f_{a}$']
#
#     for i in range(len(labels)):
#         if pattern == 0:
#             ax1.plot(depth, map_data[i], label=labels[i], color=colors[i], linestyle='-', linewidth=2)
#             ax1.set_ylabel("mAP (%)", fontsize=18)
#             for x, y in zip(depth, map_data[-1]):
#                 ax1.text(x - 0.23, y - 0.27, '%.1f' % y, fontdict={'fontsize': 14})
#         else:
#             ax1.plot(depth, r1_data[i], label=labels[i], color=colors[i], linestyle='-', linewidth=2)
#             ax1.set_ylabel("Rank-1 (%)", fontsize=18)
#             for x, y in zip(depth, r1_data[-1]):
#                 ax1.text(x - 0.25, y - 0.05, '%.1f' % y, fontdict={'fontsize': 14})
#
#     ax1.set_xlabel("depth", fontsize=18)
#     ax1.set_xticks(list(depth))
#     ax1.tick_params(axis='both', labelsize=10)
#     # ax1.legend(loc='upper right', fontsize=10)
#     ax1.grid(True)
#
#     for i in range(len(labels)):
#         if pattern == 1:
#             ax2.plot(depth, map_data[i], label=labels[i], color=colors[i], linestyle='-', linewidth=2)
#             ax2.set_ylabel("mAP (%)", fontsize=18)
#             for x, y in zip(depth, map_data[-1]):
#                 ax2.text(x - 0.23, y - 0.27, '%.1f' % y, fontdict={'fontsize': 14})
#         else:
#             ax2.plot(depth, r1_data[i], label=labels[i], color=colors[i], linestyle='-', linewidth=2)
#             ax2.set_ylabel("Rank-1 (%)", fontsize=18)
#             for x, y in zip(depth, r1_data[-1]):
#                 ax2.text(x - 0.20, y - 0.1, '%.1f' % y, fontdict={'fontsize': 14})
#
#     ax2.set_xlabel("depth", fontsize=18)
#     ax2.set_xticks(list(depth))
#     ax2.tick_params(axis='both', labelsize=10)
#     ax2.legend(loc='upper right', fontsize=10)
#     ax2.grid(True)
#
#     plt.tight_layout()
#     plt.savefig(dataset + dict[pattern] + '.pdf', bbox_inches='tight')
#     plt.show()
#
# filename_s = 'Market1501_mAP_R1.txt'
# depth_effect(filename_s, pattern=0)
from matplotlib import pyplot as plt

def get_data(str):
    str = str.replace('\n', '').split('\t')
    data = []
    for item in str[1:]:
        data.append(float(item))
    return data

def get_all_data(txt):
    data = []
    for item in txt:
        data.append(get_data(item))
    return data

def txt_process(filename_s, choice=1):
    with open(filename_s) as f:
        lines = f.readlines()
    if choice == 0:
        txt = lines[1:8]
    else:
        txt = lines[9:]
    data = get_all_data(txt)
    return data

def depth_effect(filename_s,filename_l):
    dataset = filename_s.split('_')[0]
    map_data = txt_process(filename_s, choice=0)
    r1_data = txt_process(filename_s, choice=1)
    depth = range(1, 7)
    dict = {0: "mAP", 1: "Rank-1"}
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(12, 12), dpi=250)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    labels = ['$\hat{f_c}$', "$f_{(c,0)}'$", "$f_{(c,L)}'$", '$\hat{f_t}$', "$f_{(t,0)}'$", "$f_{(t,L)}'$", '$f_{a}$']

    for i in range(len(labels)):

        ax1[0].plot(depth, map_data[i], label=labels[i], color=colors[i], linestyle='-', linewidth=2)
        ax1[0].set_ylabel("mAP (%)", fontsize=15)

        for x, y in zip(depth, map_data[-1]):
            ax1[0].text(x - 0.25, y - 0.20, '%.1f' % y, fontdict={'fontsize': 14})
        ax1[0].set_title("Market1501  mAP", fontsize=18)

        ax1[1].plot(depth, r1_data[i], label=labels[i], color=colors[i], linestyle='-', linewidth=2)
        ax1[1].set_ylabel("Rank-1 (%)", fontsize=15)

        for x, y in zip(depth, r1_data[-1]):
            ax1[1].text(x - 0.25, y - 0.04, '%.1f' % y, fontdict={'fontsize': 14})
        ax1[1].set_title("Market1501  Rank-1", fontsize=18)
    ax1[0].grid(True)
    ax1[1].grid(True)
    map_data = txt_process(filename_l, choice=0)
    r1_data = txt_process(filename_l, choice=1)
    for i in range(len(labels)):
        ax2[0].plot(depth, map_data[i], label=labels[i], color=colors[i], linestyle='-', linewidth=2)
        ax2[0].set_ylabel("mAP (%)", fontsize=18)

        for x, y in zip(depth, map_data[-1]):
            ax2[0].text(x - 0.25, y - 0.27, '%.1f' % y, fontdict={'fontsize': 14})
        ax2[0].set_title("MSMT17  mAP", fontsize=18)
        ax2[1].plot(depth, r1_data[i], label=labels[i], color=colors[i], linestyle='-', linewidth=2)
        ax2[1].set_ylabel("Rank-1 (%)", fontsize=18)

        for x, y in zip(depth, r1_data[-1]):
            ax2[1].text(x - 0.25, y - 0.05, '%.1f' % y, fontdict={'fontsize': 14})
        ax2[1].set_title("MSMT17  Rank-1", fontsize=18)
    ax2[0].grid(True)
    ax2[1].grid(True)
    fig.text(0.5, 0.04, 'depth', ha='center', fontsize=18)
    handles, labels = ax1[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize=18,bbox_to_anchor=(0.5, 1.0), bbox_transform=fig.transFigure, ncol=7)
    fig.tight_layout(pad=5.0)
    plt.savefig( 'depth.pdf', bbox_inches='tight')
    plt.show()

filename_s = 'Market1501_mAP_R1.txt'
filename_l = 'MSMT_mAP_R1.txt'
depth_effect(filename_s,filename_l)
