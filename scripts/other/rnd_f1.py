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
    name_exp = f'exp/rnd/yolo_s_{datetime.datetime.now().strftime("%d%m%Y")}_person'

    with Live(dir=name_exp, save_dvc_exp=True) as live:
        pretrain_from_hub = False
        retrain_user_model = False
        type_model = 'custom'
        num_epochs = 100
        gpu = 0
        params_map = {
            'gpu': gpu,
            'path_to_img_train':    '/media/alex/DAtA4/Datasets/coco/my_dataset/train',
            'path_to_labels_train': '/media/alex/DAtA4/Datasets/coco/labelstrain',
            'path_to_img_val':      '/media/alex/DAtA4/Datasets/coco/my_dataset/val',
            'path_to_labels_val':   '/media/alex/DAtA4/Datasets/coco/my_dataset/labels_val/val.json',
            'path_to_img_test':     '/media/alex/DAtA4/Datasets/coco/my_dataset/test',
            'path_to_labels_test':  '/media/alex/DAtA4/Datasets/coco/my_dataset/labels_test/test.json',
            'pretrain_from_hub':     pretrain_from_hub,
            'save_model':            True,
            'path_model':            '',
            'retrain_user_model':    retrain_user_model,
            'type_model':            type_model,
            'num_epochs':            num_epochs

        }

        live.log_param("type_model", params_map['type_model'])
        live.log_param("num_epochs (val)", params_map['num_epochs'])
        live.log_param("pretrain_from_hub (val)", params_map['pretrain_from_hub'])
        live.log_param("retrain_user_model (val)", params_map['retrain_user_model'])

        # p = [1_000, 10_000, 30_000]
        N_train = len(os.listdir('/media/alex/DAtA4/Datasets/coco/my_dataset/train'))
        # N_train = len(os.listdir('/home/neptun/PycharmProjects/datasets/coco/my_dataset/train'))
        # n_rnd = [N_train // 64, N_train // 32, N_train // 16, N_train // 8, N_train // 4, N_train // 2, N_train]
        n_rnd = [8_000, 12_000]
        # n_rnd = [5_000, 10_000, 15_000, 20_000]
        live.log_param("n_rnd", n_rnd)

        # n_rnd = [N_train // 64]
        print(n_rnd)
        start = datetime.datetime.now()
        print(start)
        told = start
        k = 3
        L = []
        for j in range(k):
            for i in n_rnd:
                make_file(i,
                          path_to_json_train='/media/alex/DAtA4/Datasets/coco/my_dataset/labels_train/train.json',
                          path_to_out='/media/alex/DAtA4/Datasets/coco/labelstrain/first.json')

                out = mAP(params_map)
                metrics_test, model = out['metrics_test'], out['model']
                live.log_metric('test/mAP50', metrics_test[2])
                live.log_metric('test/P', metrics_test[0])
                live.log_metric('test/R', metrics_test[1])
                # live.log_metric('train/mAP50', metrics_train[2])
                # live.log_metric('train/P', metrics_train[0])
                # live.log_metric('train/R', metrics_train[1])

                if type_model == 'custom':
                    root_p = params_map['path_to_img_train']
                    files = os.listdir(root_p)
                    files = [os.path.join(root_p, x) for x in files]
                    list_train = random.sample(files, k=5)
                    save_images_detect(live, model,
                                       list_train,
                                       i, gpu)

                live.next_step()


            t = datetime.datetime.now()
            print(t, t-told)
            told = t

