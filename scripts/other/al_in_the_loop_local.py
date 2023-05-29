from urllib import request, parse
import json
import os
from createrandomlabels import make_file
from list_to_cocofile import write_json
from scripts.detection.train import train_api
from scripts.detection.eval import eval
import torch
import datetime

def al(add, path_model=''):
    url = 'http://127.0.0.1:5000/active_learning'
    params = {
        'gpu': '0',
        'path_to_img_train': '/media/alex/DAtA4/Datasets/coco/my_dataset/train',
        'path_to_labels_train': '/media/alex/DAtA4/Datasets/coco/for_al',
        'path_to_img_val': '/media/alex/DAtA4/Datasets/coco/my_dataset/val',
        'path_to_labels_val': '/media/alex/DAtA4/Datasets/coco/my_dataset/labels_val/val.json',
        'pretrain_from_hub': True,
        'save_model': False,
        'path_model': path_model,
        'retrain_user_model': False,
        'add': add,
        'batch_unlabeled': -1,
        'use_val_test_in_train': True,
        'bbox_selection_policy': 'min',
        'quantile_min': 0,
        'quantile_max': 0.3,
        'type_model': 'custom',
        'num_epochs': 5
    }

    # querystring = parse.urlencode(params)
    #
    # u = request.urlopen(url + '?' + querystring)
    # resp = u.read()
    # out = json.loads(resp.decode('utf-8'))['data']
    out = train_api(**params)
    return out

def mAP():
    url = 'http://127.0.0.1:5000/eval'
    params = {
        'gpu': '0',
        'path_to_img_train': '/media/alex/DAtA4/Datasets/coco/my_dataset/train',
        'path_to_labels_train': '/media/alex/DAtA4/Datasets/coco/for_al',
        'path_to_img_val': '/media/alex/DAtA4/Datasets/coco/my_dataset/val',
        'path_to_labels_val': '/media/alex/DAtA4/Datasets/coco/my_dataset/labels_val/val.json',
        'path_to_img_test': '/media/alex/DAtA4/Datasets/coco/my_dataset/test',
        'path_to_labels_test': '/media/alex/DAtA4/Datasets/coco/my_dataset/labels_test/test.json',
        'pretrain_from_hub': True,
        'save_model': True,
        'path_model': '',
        'retrain_user_model': False,
        'type_model': 'custom',
        'num_epochs': 5

    }

    # querystring = parse.urlencode(params)
    #
    # u = request.urlopen(url + '?' + querystring)
    # resp = u.read()
    # out = json.loads(resp.decode('utf-8'))
    out = eval(**params)
    return (out['mAP(0.5:0.95)'], out['model'])

if __name__ == '__main__':
    L = []
    path_to_img_train = '/media/alex/DAtA4/Datasets/coco/my_dataset/train'
    path_to_labels_train = '/media/alex/DAtA4/Datasets/coco/for_al'
    path_to_json_train = '/media/alex/DAtA4/Datasets/coco/my_dataset/labels_train/train.json'

    N_train = len(os.listdir(path_to_img_train))
    n_al = [N_train // 64, N_train // 32, N_train // 16, N_train // 8, N_train // 4, N_train // 2]
    print(n_al)
    start = datetime.datetime.now()
    print(start)
    told = start

    for i in range(1):
        files_in_labels = os.listdir(path_to_labels_train)
        for file in files_in_labels:
            os.remove(os.path.join(path_to_labels_train, file))
        make_file(n_al[0],
                  path_to_json_train=path_to_json_train,
                  path_to_out=os.path.join(path_to_labels_train, 'first.json'))
        a = []
        for kk in range(len(n_al)):
            out = mAP()
            f, path_model = out[0], out[1]
            a.append(f)
            print('mAP', n_al[kk], f)
            if kk != len(n_al) - 1:
                step = al(n_al[kk], path_model)
                write_json(step,
                           kk,
                           path_to_out=path_to_labels_train,
                           full_train_json=path_to_json_train)
            t = datetime.datetime.now()
            print(t, t - told)
            told = t

        print(a)
        L.append(a)
    print(L)
