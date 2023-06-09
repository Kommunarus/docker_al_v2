import random
from urllib import request, parse
import json
import os
from createrandomlabels import make_file
from list_to_cocofile import write_json
from scripts.detection.train import train_api
from scripts.detection.eval import eval
from scripts.detection.custom_dataset_yolo import save_images_detect

import torch
import datetime

from dvclive import Live
import numpy as np
import matplotlib.pyplot as plt
import shutil


def al(params):
    url = 'http://127.0.0.1:5000/active_learning'
    # querystring = parse.urlencode(params)
    #
    # u = request.urlopen(url + '?' + querystring)
    # resp = u.read()
    # out = json.loads(resp.decode('utf-8'))['data']
    out = train_api(**params)
    return out['data'], out['all_value'], out['path_to_img']

def mAP(params):
    url = 'http://127.0.0.1:5000/eval'


    # querystring = parse.urlencode(params)
    #
    # u = request.urlopen(url + '?' + querystring)
    # resp = u.read()
    # out = json.loads(resp.decode('utf-8'))
    return eval(**params)

if __name__ == '__main__':
    number_experiment = 3
    class_data = 17

    f1 = f'exp/{number_experiment}/al'
    name_e = f'yolo_s_{datetime.datetime.now().strftime("%d%m%Y")}'
    if os.path.exists(f1):
        all_dir = os.listdir(f1)
        k = sum([1 for name in all_dir if name.find(name_e) > -1])
    else:
        k = 0

    name_exp = f'exp/{number_experiment}/al/yolo_s_{datetime.datetime.now().strftime("%d%m%Y")}_{k}'
    name_my_dataset = 'my_dataset_cat'

    with Live(dir=name_exp, save_dvc_exp=True) as live:
        type_model = 'custom'
        # type_model = 'fasterrcnn'
        num_epochs = 100
        retrain_user_model = False
        pretrain_from_hub = False
        batch_unlabeled = -1
        bbox_selection_policy = 'mean'
        selection_min = True
        selection_max = False
        quantile_min = 0.7
        quantile_max = 1
        gpu = 1
        params_map = {
            'gpu': gpu,
            'path_to_img_train': f'/media/alex/DAtA4/Datasets/coco/{name_my_dataset}/train',
            'path_to_labels_train': '/media/alex/DAtA4/Datasets/coco/for_al',
            'path_to_img_val': f'/media/alex/DAtA4/Datasets/coco/{name_my_dataset}/val',
            'path_to_labels_val': f'/media/alex/DAtA4/Datasets/coco/{name_my_dataset}/labels_val/val.json',
            'path_to_img_test': f'/media/alex/DAtA4/Datasets/coco/{name_my_dataset}/test',
            'path_to_labels_test': f'/media/alex/DAtA4/Datasets/coco/{name_my_dataset}/labels_test/test.json',
            'pretrain_from_hub': pretrain_from_hub,
            'save_model': True,
            'path_model': '',
            'retrain_user_model': retrain_user_model,
            'type_model': type_model,
            'num_epochs': num_epochs

        }
        live.log_param("class dataset", 'cat')
        live.log_param("type dataset", 'balance')

        live.log_param("type_model", params_map['type_model'])
        live.log_param("num_epochs (val)", params_map['num_epochs'])
        live.log_param("pretrain_from_hub (val)", params_map['pretrain_from_hub'])
        live.log_param("retrain_user_model (val)", params_map['retrain_user_model'])

        params_al = {
            'gpu': gpu,
            'path_to_img_train': f'/media/alex/DAtA4/Datasets/coco/{name_my_dataset}/train',
            'path_to_labels_train': '/media/alex/DAtA4/Datasets/coco/for_al',
            'path_to_img_val': f'/media/alex/DAtA4/Datasets/coco/{name_my_dataset}/val',
            'path_to_labels_val': f'/media/alex/DAtA4/Datasets/coco/{name_my_dataset}/labels_val/val.json',
            'pretrain_from_hub': pretrain_from_hub,
            'save_model': False,
            'retrain_user_model': retrain_user_model,
            'batch_unlabeled': batch_unlabeled,
            'use_val_test_in_train': True,
            'bbox_selection_policy': bbox_selection_policy,
            'quantile_min': quantile_min,
            'quantile_max': quantile_max,
            'selection_min': selection_min,
            'selection_max': selection_max,
            'type_model': type_model,
            'num_epochs': num_epochs
        }

        live.log_param("quantile_min (al)", params_al['quantile_min'])
        live.log_param("quantile_max (al)", params_al['quantile_max'])
        live.log_param("selection_min (al)", params_al['selection_min'])
        live.log_param("selection_max (al)", params_al['selection_max'])
        live.log_param("batch_unlabeled (al)", params_al['batch_unlabeled'])
        live.log_param("bbox_selection_policy (al)", params_al['bbox_selection_policy'])


        L = []
        path_to_img_train = f'/media/alex/DAtA4/Datasets/coco/{name_my_dataset}/train'
        path_to_labels_train = '/media/alex/DAtA4/Datasets/coco/for_al'
        path_to_json_train = f'/media/alex/DAtA4/Datasets/coco/{name_my_dataset}/labels_train/train.json'

        N_train = len(os.listdir(path_to_img_train))
        n_al = [N_train // 32] + [N_train // 32] * 20
        # n_al = [N_train // 64, N_train // 32, N_train // 16, N_train // 8, N_train // 4]
        # n_al = [5_000, ] + [1_000, ] * 10
        live.log_param('n_al', n_al)

        # n_al = [N_train // 64, N_train // 32, N_train // 16]
        print(n_al)
        start = datetime.datetime.now()
        print(start)
        told = start

        for i in range(15):
            datapoints = []

            files_in_labels = os.listdir(path_to_labels_train)
            for file in files_in_labels:
                os.remove(os.path.join(path_to_labels_train, file))
            count_good_image = make_file(n_al[0],
                  path_to_json_train=path_to_json_train,
                  path_to_out=os.path.join(path_to_labels_train, 'first.json'), class_data=class_data)
            live.log_metric('files/count_good_image', count_good_image)
            live.log_metric('files/raw_image', n_al[0])

            for kk in range(1, len(n_al)):
                out = mAP(params_map)
                metrics_test, model = out['metrics_test'], out['model']
                if type_model == 'custom':
                    live.log_metric('test/mAP50', metrics_test[2])
                    # live.log_metric('test/P', metrics_test[0])
                    # live.log_metric('test/R', metrics_test[1])

                    datapoints.append({'samples': sum(n_al[:kk]), 'map': metrics_test[2]})

                if kk != len(n_al) - 1:
                    params_al['path_model'] = model
                    params_al['add'] = n_al[kk]
                    delta, all_value, path_to_img = al(params_al)
                    count_good_image = write_json(delta,
                                                  kk,
                                                  path_to_out=path_to_labels_train,
                                                  class_data=class_data,
                                                  full_train_json=path_to_json_train)

                    fig, ax = plt.subplots()
                    ax.hist(np.array(all_value), bins=50)
                    fig.canvas.draw()
                    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                    live.log_image(f"distribution_{i}_{kk}.png", data)
                    live.log_metric('files/count_good_image', count_good_image)
                    live.log_metric('files/raw_image', n_al[kk])

                    if type_model == 'custom':
                        shutil.rmtree(path_to_img)

                t = datetime.datetime.now()
                print(t, t - told)
                live.log_metric('time', (t - told).seconds/60)
                told = t

                live.next_step()

            live.log_plot(
                f'map50_{i+1}',
                datapoints,
                x="samples",
                y="map",
                template="scatter",
                title=f"map50 for al. line {i+1}",
                y_label="samples",
                x_label="map50"
            )
