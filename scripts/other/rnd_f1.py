from urllib import request, parse
import json
from createrandomlabels import make_file
import matplotlib.pyplot as plt
from scripts.detection.eval import  eval
import os
import datetime


def mAP():
    url = 'http://127.0.0.1:5000/eval'
    params = {
        'gpu': '0',
        'path_to_img_train': '/ds/coco/my_dataset/train',
        'path_to_labels_train': '/ds/coco/labelstrain/first.json',
        'path_to_img_val': '/ds/coco/my_dataset/val',
        'path_to_labels_val': '/ds/coco/my_dataset/labels_val/val.json',
        'path_to_img_test': '/ds/coco/my_dataset/test',
        'path_to_labels_test': '/ds/coco/my_dataset/labels_test/test.json',
        'pretrain_from_hub': 'F',
        'save_model': 'F',
        'path_model': '',
        'retrain_user_model': 'F',
        'type_model': 'custom',
        'num_epochs': '50',

    }

    querystring = parse.urlencode(params)

    u = request.urlopen(url + '?' + querystring)
    resp = u.read()
    out = json.loads(resp.decode('utf-8'))
    res = out['mAP(0.5:0.95)']
    return res
    # path_to_img_train       = '/media/alex/DAtA2/Datasets/coco/my_dataset/train'
    # path_to_labels_train    = '/media/alex/DAtA2/Datasets/coco/labelstrain/first.json'
    # path_to_img_val         = '/media/alex/DAtA2/Datasets/coco/my_dataset/val'
    # path_to_labels_val      = '/media/alex/DAtA2/Datasets/coco/my_dataset/labels_val/val.json'
    # path_to_img_test        = '/media/alex/DAtA2/Datasets/coco/my_dataset/test'
    # path_to_labels_test     = '/media/alex/DAtA2/Datasets/coco/my_dataset/labels_test/test.json'
    # path_to_labels_train = '/home/neptun/PycharmProjects/datasets/coco/labelstrain/first.json'
    # path_to_img_train = '/home/neptun/PycharmProjects/datasets/coco/train2017'
    # path_to_labels_val = '/home/neptun/PycharmProjects/datasets/coco/labelsval/val.json'
    # path_to_img_val = '/home/neptun/PycharmProjects/datasets/coco/val2017'
    # device_rest = '0'
    # path_do_dir_model = '../../weight'
    # pretrain = True
    # return eval(path_to_img_train, path_to_labels_train,
    #             path_to_img_val, path_to_labels_val,
    #             path_to_img_test, path_to_labels_test, save_model, path_do_dir_model,
    #             device_rest, pretrain)

if __name__ == '__main__':
    # p = [1_000, 10_000, 30_000]
    N_train = len(os.listdir('/home/neptun/PycharmProjects/datasets/coco/my_dataset/train'))
    # n_rnd = [N_train // 64]
    n_rnd = [N_train // 64, N_train // 32, N_train // 16, N_train // 8, N_train // 4, N_train // 2, N_train]
    start = datetime.datetime.now()
    print(start)
    told = start
    k = 5
    L = []
    for i in n_rnd:
        mean = 0
        m = []
        for j in range(k):
            make_file(i,
                      path_to_json_train='/home/neptun/PycharmProjects/datasets/coco/my_dataset/labels_train/train.json',
                      path_to_out='/home/neptun/PycharmProjects/datasets/coco/labelstrain/first.json')
            # make_file(i,
            #           path_to_json_train='/media/alex/DAtA3/Datasets/coco/my_dataset/labels_train/train.json',
            #           path_to_out='/media/alex/DAtA3/Datasets/coco/labelstrain/first.json')
            f = mAP()
            print(f)
            mean += f
            m.append(f)
        L.append(mean/k)
        plt.scatter([i]*k, m)
        print(i, m)
        t = datetime.datetime.now()
        print(t, t-told)
        told = t

    plt.plot(n_rnd, L)
    plt.grid(True)
    plt.show()
    # print(L)