from urllib import request, parse
import json
import os
from createrandomlabels import make_file
from list_to_cocofile import write_json
from scripts.detection.train import train_api
from scripts.detection.eval import eval
import torch
import datetime

from dvclive import Live
import numpy as np


def al(params):
    url = 'http://127.0.0.1:5000/active_learning'
    # querystring = parse.urlencode(params)
    #
    # u = request.urlopen(url + '?' + querystring)
    # resp = u.read()
    # out = json.loads(resp.decode('utf-8'))['data']
    out = train_api(**params)
    return out['data'], out['all_value']

def mAP(params):
    url = 'http://127.0.0.1:5000/eval'


    # querystring = parse.urlencode(params)
    #
    # u = request.urlopen(url + '?' + querystring)
    # resp = u.read()
    # out = json.loads(resp.decode('utf-8'))
    out = eval(**params)
    return (out['mAP(0.5:0.95)'], out['model'])

if __name__ == '__main__':
    name_exp = 'exp'
    with Live(dir=name_exp ,save_dvc_exp=True) as live:
        # type_model = 'custom'
        type_model = 'fasterrcnn'
        num_epochs = 20
        retrain_user_model = False
        pretrain_from_hub = False
        params_map = {
            'gpu': '0',
            'path_to_img_train': '/media/alex/DAtA4/Datasets/coco/my_dataset/train',
            'path_to_labels_train': '/media/alex/DAtA4/Datasets/coco/for_al',
            'path_to_img_val': '/media/alex/DAtA4/Datasets/coco/my_dataset/val',
            'path_to_labels_val': '/media/alex/DAtA4/Datasets/coco/my_dataset/labels_val/val.json',
            'path_to_img_test': '/media/alex/DAtA4/Datasets/coco/my_dataset/test',
            'path_to_labels_test': '/media/alex/DAtA4/Datasets/coco/my_dataset/labels_test/test.json',
            'pretrain_from_hub': pretrain_from_hub,
            'save_model': True,
            'path_model': '',
            'retrain_user_model': retrain_user_model,
            'type_model': type_model,
            'num_epochs': num_epochs

        }
        live.log_param("type_model (val)", params_map['type_model'])
        live.log_param("num_epochs (val)", params_map['num_epochs'])
        live.log_param("pretrain_from_hub (val)", params_map['pretrain_from_hub'])
        live.log_param("retrain_user_model (val)", params_map['retrain_user_model'])

        params_al = {
            'gpu': '0',
            'path_to_img_train': '/media/alex/DAtA4/Datasets/coco/my_dataset/train',
            'path_to_labels_train': '/media/alex/DAtA4/Datasets/coco/for_al',
            'path_to_img_val': '/media/alex/DAtA4/Datasets/coco/my_dataset/val',
            'path_to_labels_val': '/media/alex/DAtA4/Datasets/coco/my_dataset/labels_val/val.json',
            'pretrain_from_hub': pretrain_from_hub,
            'save_model': False,
            'retrain_user_model': retrain_user_model,
            'batch_unlabeled': -1,
            'use_val_test_in_train': True,
            'bbox_selection_policy': 'min',
            'quantile_min': 0,
            'quantile_max': 0.3,
            'type_model': type_model,
            'num_epochs': num_epochs
        }

        live.log_param("type_model (al)", params_al['type_model'])
        live.log_param("quantile_min (al)", params_al['quantile_min'])
        live.log_param("quantile_max (al)", params_al['quantile_max'])
        live.log_param("batch_unlabeled (al)", params_al['batch_unlabeled'])
        live.log_param("bbox_selection_policy (al)", params_al['bbox_selection_policy'])


        L = []
        path_to_img_train = '/media/alex/DAtA4/Datasets/coco/my_dataset/train'
        path_to_labels_train = '/media/alex/DAtA4/Datasets/coco/for_al'
        path_to_json_train = '/media/alex/DAtA4/Datasets/coco/my_dataset/labels_train/train.json'

        N_train = len(os.listdir(path_to_img_train))
        # n_al = [N_train // 64, N_train // 32, N_train // 16, N_train // 8, N_train // 4, N_train // 2]
        n_al = [N_train // 64, N_train // 32, N_train // 16]
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
                out = mAP(params_map)
                f, path_model = out[0], out[1]
                a.append(f)
                print('mAP', n_al[kk], f)
                live.log_metric(f'mAP_step_{kk}', f)
                if kk != len(n_al) - 1:
                    params_al['path_model'] = path_model
                    params_al['add'] = n_al[kk]
                    delta, all_value = al(params_al)
                    write_json(delta,
                               kk,
                               path_to_out=path_to_labels_train,
                               full_train_json=path_to_json_train)
                    hist, bins = np.histogram(np.array(all_value))
                    datapoints = []
                    for v, x in zip(hist, bins[1:]) :
                        datapoints.append({'p': x, 'n': v})
                    live.log_plot('p inference', datapoints, x='p', y='n')
                t = datetime.datetime.now()
                print(t, t - told)
                live.log_metric(f'time_work_step_{kk}', (t - told).seconds/60)
                told = t

                live.next_step()

            print(a)
            L.append(a)
        print(L)
