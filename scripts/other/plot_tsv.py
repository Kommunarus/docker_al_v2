import matplotlib.pyplot as plt

# [528, 1057, 2114, 4228, 8456, 16913, 33827]
listik = [
    [
        'map50 test',
        './exp/rnd/yolo_s_05062023_person/plots/metrics/test/mAP50.tsv',
        './exp/al/yolo_s_05062023_person/plots/metrics/test/mAP50.tsv',
    ],
]

for exp in listik:
    path_to_tsv = exp[1]

    steps = []
    values = []
    case = 0

    with open(path_to_tsv) as f:
        first = f.readline()
        for row in f.readlines():
            arr = row.strip().split()
            steps.append(int(arr[0]))
            values.append(float(arr[1]))

    if case == 0:
        x_rnd = [5_000, 10_000, 15_000, 20_000]
        r = len(x_rnd)
        N = len(steps)
        k = N // r + 1
        vn = []
        for i in range(k):
            vn.append(values[i*r: (i+1)*r])

        for i, v in enumerate(vn):
            plt.scatter(x_rnd[:len(v)], v)
        # plt.show()

    path_to_tsv2 = exp[2]

    steps = []
    values = []
    case = 1

    with open(path_to_tsv2) as f:
        first = f.readline()
        for row in f.readlines():
            arr = row.strip().split()
            steps.append(int(arr[0]))
            values.append(float(arr[1]))

    if case == 1:
        x_al = [5_000, ] + [1_000, ] * 10
        x_al = [sum(x_al[:i]) for i in range(1, len(x_al))]
        r = len(x_al)
        N = len(steps)
        k = N // r + 1
        vn = []
        for i in range(k):
            vn.append(values[i * r: (i + 1) * r])

        for i, v in enumerate(vn):
            plt.plot(x_al[:len(v)], v)
        plt.title(exp[0])


listik = [
    [
        'map50 test',
        './exp/rnd/yolo_s_06062023_person/plots/metrics/test/mAP50.tsv',
    ],
]

for exp in listik:
    path_to_tsv = exp[1]

    steps = []
    values = []
    case = 0

    with open(path_to_tsv) as f:
        first = f.readline()
        for row in f.readlines():
            arr = row.strip().split()
            steps.append(int(arr[0]))
            values.append(float(arr[1]))

    if case == 0:
        x_rnd = [8_000, 12_000]
        r = len(x_rnd)
        N = len(steps)
        k = N // r + 1
        vn = []
        for i in range(k):
            vn.append(values[i*r: (i+1)*r])

        for i, v in enumerate(vn):
            plt.scatter(x_rnd[:len(v)], v)
        # plt.show()

plt.grid()
plt.show()