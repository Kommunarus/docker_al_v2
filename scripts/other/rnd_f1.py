from urllib import request, parse
import json
from createrandomlabels import make_file
import matplotlib.pyplot as plt
from scripts.detection.eval import  eval
import os
import datetime
from dvclive import Live
import random
from scripts.detection.custom_dataset_yolo import save_images_detect


def mAP(params):
    return eval(**params)

if __name__ == '__main__':
    number_experiment = 2
    name_exp = f'exp/{number_experiment}/rnd/yolo_s_{datetime.datetime.now().strftime("%d%m%Y")}'
    name_my_dataset = 'my_dataset_person_and_other'

    with Live(dir=name_exp, save_dvc_exp=True) as live:
        pretrain_from_hub = False
        retrain_user_model = False
        type_model = 'custom'
        num_epochs = 100
        gpu = 0
        params_map = {
            'gpu': gpu,
            'path_to_img_train':    f'/media/alex/DAtA4/Datasets/coco/{name_my_dataset}/train',
            'path_to_labels_train': '/media/alex/DAtA4/Datasets/coco/labelstrain',
            'path_to_img_val':      f'/media/alex/DAtA4/Datasets/coco/{name_my_dataset}/val',
            'path_to_labels_val':   f'/media/alex/DAtA4/Datasets/coco/{name_my_dataset}/labels_val/val.json',
            'path_to_img_test':     f'/media/alex/DAtA4/Datasets/coco/{name_my_dataset}/test',
            'path_to_labels_test':  f'/media/alex/DAtA4/Datasets/coco/{name_my_dataset}/labels_test/test.json',
            'pretrain_from_hub':     pretrain_from_hub,
            'save_model':            True,
            'path_model':            '',
            'retrain_user_model':    retrain_user_model,
            'type_model':            type_model,
            'num_epochs':            num_epochs

        }

        live.log_param("class dataset", 'person')
        live.log_param("type dataset", 'disbalance')
        live.log_param("length train part", len(os.listdir(params_map['path_to_img_train'])))
        live.log_param("length val part", len(os.listdir(params_map['path_to_img_val'])))
        live.log_param("length test part", len(os.listdir(params_map['path_to_img_test'])))

        live.log_param("type_model", params_map['type_model'])
        live.log_param("num_epochs (val)", params_map['num_epochs'])
        live.log_param("pretrain_from_hub (val)", params_map['pretrain_from_hub'])
        live.log_param("retrain_user_model (val)", params_map['retrain_user_model'])

        # p = [1_000, 10_000, 30_000]
        N_train = len(os.listdir(f'/media/alex/DAtA4/Datasets/coco/{name_my_dataset}/train'))
        # N_train = len(os.listdir('/home/neptun/PycharmProjects/datasets/coco/my_dataset/train'))
        # n_rnd = [N_train // 64, N_train // 32, N_train // 16]
        n_rnd = [N_train // 32, N_train // 16, N_train // 8, N_train // 4, N_train // 2, N_train]
        # n_rnd = [8_000, 12_000]
        # n_rnd = [5_000, 10_000, 15_000, 20_000]
        live.log_param("n_rnd", n_rnd)

        # n_rnd = [N_train // 64]
        print(n_rnd)
        start = datetime.datetime.now()
        print(start)
        told = start
        k = 3
        L = []
        for nn, j in enumerate(range(k)):
            datapoints = []


            for i in n_rnd:
                count_good_image = make_file(i,
                      path_to_json_train=f'/media/alex/DAtA4/Datasets/coco/{name_my_dataset}/labels_train/train.json',
                      path_to_out='/media/alex/DAtA4/Datasets/coco/labelstrain/first.json')
                live.log_metric('files/count_good_image', count_good_image)
                live.log_metric('files/raw_image', i)

                out = mAP(params_map)

                metrics_test, model = out['metrics_test'], out['model']
                live.log_metric('test/mAP50', metrics_test[2])
                live.log_metric('test/P', metrics_test[0])
                live.log_metric('test/R', metrics_test[1])
                datapoints.append({'samples': i, 'map': metrics_test[2]})

                live.next_step()

            live.log_plot(
                f'map50_{nn+1}',
                datapoints,
                x="samples",
                y="map",
                template="scatter",
                title=f"map50 for al. line {nn+1}",
                y_label="samples",
                x_label="map50"
            )
            t = datetime.datetime.now()
            print(t, t-told)
            told = t

