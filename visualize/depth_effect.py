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


def txt_process(filename, choice=1):
    f = open(filename)
    line = f.readlines()  # 调用文件的 readline()方法
    if choice == 0:
        txt = line[1:8]
    else:
        txt = line[9:]
    data = get_all_data(txt)
    f.close()
    return data


def depth_effect(filename, pattern=0):
    dataset = filename.split('_')[0]
    map = txt_process(filename, choice=0)
    R1 = txt_process(filename, choice=1)
    depth = range(1, 9)
    plt.figure(dpi=350)
    dict = {0: "mAP", 1: "Rank-1"}
    # figure_name = "Depth's effect on " + dataset
    if pattern == 0:
        plt.plot(depth, map[0], label='$\hat{f_c}$')
        plt.plot(depth, map[1], label='$\hat{f_t}$')
        plt.plot(depth, map[2], label="$f_{(c,0)}'$")
        plt.plot(depth, map[3], label="$f_{(t,0)}'$")
        plt.plot(depth, map[4], label="$f_{(c,L)}'$")
        plt.plot(depth, map[5], label="$f_{(t,L)}'$")
        plt.plot(depth, map[6], label='$f_{a}$')
        # plt.plot(depth, map[0])
        # plt.plot(depth, map[1])
        # plt.plot(depth, map[2])
        # plt.plot(depth, map[3])
        # plt.plot(depth, map[4])
        # plt.plot(depth, map[5])
        # plt.plot(depth, map[6])
        plt.ylabel("mAP (%)",fontsize=18)
        for x, y in zip(depth, map[-1]):
            plt.text(x - 0.25, y - 0.27, '%.1f' % y, fontdict={'fontsize': 12})

    else:
        plt.plot(depth, R1[0], label='$\hat{f_c}$')
        plt.plot(depth, R1[1], label='$\hat{f_t}$')
        plt.plot(depth, R1[2], label="$f_{(c,0)}'$")
        plt.plot(depth, R1[3], label="$f_{(t,0)}'$")
        plt.plot(depth, R1[4], label="$f_{(c,L)}'$")
        plt.plot(depth, R1[5], label="$f_{(t,L)}'$")
        plt.plot(depth, R1[6], label='$f_{a}$')
        # plt.plot(depth, R1[0])
        # plt.plot(depth, R1[1])
        # plt.plot(depth, R1[2])
        # plt.plot(depth, R1[3])
        # plt.plot(depth, R1[4])
        # plt.plot(depth, R1[5])
        # plt.plot(depth, R1[6])
        plt.ylabel("Rank-1 (%)",fontsize=18)
        for x, y in zip(depth, R1[-1]):
            plt.text(x - 0.25, y -0.05, '%.1f' % y, fontdict={'fontsize': 12})

    _xtick = list(depth)
    plt.xticks(_xtick[::1], rotation=0)
    plt.xlabel("depth",fontsize=18)
    # plt.title(figure_name,fontsize=18)
    plt.legend(loc=2, bbox_to_anchor=(1.01, 1.0), borderaxespad=0.,fontsize = 18)

    plt.subplots_adjust(right=0.85)
    plt.tight_layout()
    plt.savefig(dataset + dict[pattern] + '.pdf',bbox_inches = 'tight')
    plt.show()


filename = 'Market1501_mAP_R1.txt'
depth_effect(filename, pattern=1)
