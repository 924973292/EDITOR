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
    plt.figure(dpi=400)
    dict = {0: "mAP", 1: "Rank-1"}
    figure_name = "Depth's effect on" + dataset
    if pattern == 0:
        plt.plot(depth, map[0], label='$Fea_r$')
        plt.plot(depth, map[1], label='$Fea_f$')
        plt.plot(depth, map[2], label='$Fea_1$')
        plt.plot(depth, map[3], label='$Fea_2$')
        plt.plot(depth, map[4], label='$R_{mix}$')
        plt.plot(depth, map[5], label='$T_{mix}$')
        plt.plot(depth, map[6], label='$Fea_{all}$')
        plt.ylabel("mAP (%)")
        for x, y in zip(depth, map[-1]):
            plt.text(x - 0.12, y - 0.21, '%.1f' % y, fontdict={'fontsize': 8})

    else:
        plt.plot(depth, R1[0], label='$Fea_r$')
        plt.plot(depth, R1[1], label='$Fea_f$')
        plt.plot(depth, R1[2], label='$Fea_1$')
        plt.plot(depth, R1[3], label='$Fea_2$')
        plt.plot(depth, R1[4], label='$R_{mix}$')
        plt.plot(depth, R1[5], label='$T_{mix}$')
        plt.plot(depth, R1[6], label='$Fea_{all}$')
        plt.ylabel("Rank-1 (%)")
        for x, y in zip(depth, R1[-1]):
            plt.text(x - 0.15, y+0.03 , '%.1f' % y, fontdict={'fontsize': 8})

    _xtick = list(depth)
    plt.xticks(_xtick[::1], rotation=0)
    plt.xlabel("depth")
    plt.title(figure_name)
    plt.legend(loc=2, bbox_to_anchor=(1.01, 1.0), borderaxespad=0.)

    plt.subplots_adjust(right=0.85)
    plt.savefig(dataset + dict[pattern] + '.png')
    plt.show()


filename = 'Market1501_mAP_R1.txt'
depth_effect(filename, pattern=1)
