import os
from scripts.detection.engine import evaluate
import numpy as np
import torch
import json
import random
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms as t
from scripts.detection.engine import train_one_epoch
# from scripts.detection.vae import train_vae_od, find_loss_vae, plot_err_vae
from scripts.detection.classification import classification
# from scripts.detection.eval import eval
from scripts.detection.unit import Dataset_objdetect, prepare_items_od
import copy
import scripts.detection.utils as utils
# from scripts.detection.vae import get_vae_samples
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, GeneralizedRCNNTransform
from PIL import Image
import re
import uuid
from scripts.detection.unit import write_to_log
import shutil

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

import subprocess
import sys

from scripts.detection.custom_dataset_yolo import json_to_yolo

def get_transform():
    transforms = [t.ToTensor()]
    return t.Compose(transforms)

def eval_mape(model0, data_loader_test, device):
    coco_evaluator = evaluate(model0, data_loader_test, device=device)
    return {'mAP(0.5:0.95)': _summarize(coco_evaluator.coco_eval['bbox']), 'model': model0}

def train_model(pathtoimg, pathtolabelstrain,
                pathtoimgval, pathtolabelsval,
                device, num_epochs=5, pretrain=True, use_val_test=True, premodel=None):
    images_train, annotations_train = prepare_items_od(pathtoimg, pathtolabelstrain)
    write_to_log('in train {} samples'.format(len(set(images_train))))
    ds0 = Dataset_objdetect(pathtoimg, images_train, annotations_train, transforms=get_transform())
    train_dataloader = DataLoader(ds0, batch_size=8, shuffle=True, collate_fn=utils.collate_fn)

    if use_val_test:
        images_test, annotations_test = prepare_items_od(pathtoimgval, pathtolabelsval)
        write_to_log('in val {} samples'.format(len(set(images_test))))
        dataset_test = Dataset_objdetect(pathtoimgval, images_test, annotations_test, get_transform(), name='val')
        data_loader_test = DataLoader(dataset_test, batch_size=8, shuffle=False, collate_fn=utils.collate_fn)


    num_classes = 2
    best_model = None
    best_mape = 0

    if premodel is None:
        model = get_model(num_classes, pretrain)
    else:
        model = premodel
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1e-4)
    for epoch in range(num_epochs):
        # print('epoch {}'.format(epoch+1))
        train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=100)
        if use_val_test:
            outval = eval_mape(model, data_loader_test, device)
        else:
            outval = eval_mape(model, train_dataloader, device)

        mape = outval['mAP(0.5:0.95)']
        if best_mape < mape:
            best_mape = mape
            write_to_log('{}. best val mape {}'.format(epoch+1, best_mape))
        best_model = copy.deepcopy(model)
    getcwd = '/home/alex/PycharmProjects/docker_al_v2'
    if ds0.create_dataset:
        os.remove(getcwd+'/data/'+ds0.name+'.hdf5')
    if use_val_test:
        dataset_test.f5.close()
    return best_model

def train_custom(pathtoimg, pathtolabels, pathtoimgval, pathtolabelsval, pretrain_hub, num_epochs,
                 premodel=None):
    path_root, name_dir = json_to_yolo(pathtoimg, pathtolabels, pathtoimgval, pathtolabelsval)
    getcwd = '/home/alex/PycharmProjects/docker_al_v2'
    script = os.path.join(getcwd, 'custom_model/train.py')
    path_to_data_dir = os.path.join(getcwd, f'data/{name_dir}')
    path_to_data = os.path.join(path_to_data_dir, 'my.yaml')
    path_to_weight = os.path.join(getcwd, f'weight/{str(uuid.uuid4())}')

    # script = os.path.join(os.getcwd(), 'custom_model/train.py')
    # data = os.path.join(os.getcwd(), f'data/{name_dir}/my.yaml')
    # path_to_weight = os.path.join(os.getcwd(), f'weight/{str(uuid.uuid4())}')
    # os.makedirs(path_to_data_dir)
    os.makedirs(path_to_weight)
    text_yaml = '\n' + \
                f'path: {path_root + "/" + name_dir}\n' + \
                'train: train/images\n' + \
                'val: val/images\n' + \
                'test:\n' + \
                '# Classes\n' + \
                'names:\n' + \
                '   0: obj\n'
    with open(path_to_data, 'w') as f:
        f.write(text_yaml)
    if premodel is not None:
        subprocess.check_call([sys.executable, script, "--data", path_to_data, "--weights", premodel, "--img", '640',
                               '--batch-size', '16', '--epochs', str(num_epochs),
                               '--project', path_to_weight])
    else:
        weights_name = 'yolov5s'
        if pretrain_hub:
            subprocess.check_call([sys.executable, script, "--data", path_to_data, "--weights", '{}.pt'.format(weights_name),
                                   "--img", '640',
                                   '--batch-size', '16', '--epochs', str(num_epochs),
                                   '--project', path_to_weight])
        else:
            subprocess.check_call([sys.executable, script, "--data", path_to_data, "--weights", "", "--cfg",
                                   "{}.yaml".format(weights_name),
                                   "--img", '640',
                                   '--batch-size', '16', '--epochs', str(num_epochs),
                                   '--project', path_to_weight])
    shutil.rmtree(path_root + '/'+ name_dir)
    return path_to_weight + '/exp/weights/best.pt', path_to_weight

def get_model(num_classes, pretrain):
    if pretrain:
        # fasterrcnn_resnet50_fpn
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        model = fasterrcnn_resnet50_fpn(weights=weights)
    else:
        model = fasterrcnn_resnet50_fpn(weights=None)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.transform = GeneralizedRCNNTransform(min_size=640, max_size=640, image_mean=[0.485, 0.456, 0.406],
                                               image_std=[0.229, 0.224, 0.225])


    return model

def find_out_net(model, device, pathtoimg, unlabeled_data, func):
    with torch.no_grad():
        model.eval()
        dataset_train = Dataset_objdetect(pathtoimg, unlabeled_data, annotations=None, transforms=get_transform())
        train_dataloader = DataLoader(dataset_train, batch_size=32, shuffle=False, collate_fn=utils.collate_fn)
        indexs = []
        values = []
        bboxes = []
        for ep, (images, _, indx )in enumerate(train_dataloader):
            # print('epoch {}/{}'.format(ep, len(train_dataloader)))
            images = list(img.to(device) for img in images)
            outputs = model(images)
            prob = [x['scores'].tolist() for x in outputs]
            confidence = []
            for b_row in prob:
                if len(b_row) == 0:
                    confidence.append(0)
                else:
                    dd = []
                    for s in b_row:
                        dd.append(s)

                    p1 = func(dd)
                    confidence.append(p1)
            boxes = [x['boxes'].tolist() for x in outputs]

            indexs += [x for x in indx]
            values += confidence
            bboxes = bboxes + boxes
        if dataset_train.create_dataset:
            os.remove('../../data/'+dataset_train.name+'.hdf5')

    return indexs, values, bboxes

def find_out_net_custom(path_model, pathtoimg, unlabeled_data, func):
    # python  detect.py --weights  yolov5s.pt --source   path/
    getcwd = '/home/alex/PycharmProjects/docker_al_v2'
    script = getcwd + '/custom_model/detect.py'

    data_file = getcwd + f'/data/{str(uuid.uuid4())}.txt'
    with open(data_file, 'w') as list_file:
        for f in unlabeled_data:
            list_file.write('{}/{}\n'.format(pathtoimg, f))

    path_to_out = getcwd + f'/data/{str(uuid.uuid4())}'
    os.makedirs(path_to_out)

    subprocess.check_call([sys.executable, script, "--source", data_file, "--weights", path_model, "--img", '640',
                           '--project', path_to_out, '--save-txt', '--agnostic-nms', '--nosave', '--save-conf'])

    indexs = []
    values = []
    for indx, file_img in enumerate(unlabeled_data):
        # indexs.append(indx)

        name = file_img.split('.')[0]
        name_txt = os.path.join(path_to_out, 'exp', 'labels', name + '.txt')
        if os.path.exists(name_txt):
            with open(name_txt) as f_t:
                prob = [float(x.strip().split(' ')[5]) for x in f_t.readlines()]
            p1 = func(prob)
            indexs += [indx, ]
            values += [p1, ]
        else:
            indexs += [indx, ]
            values += [0, ]


    shutil.rmtree(path_to_out)
    os.remove(data_file)
    return indexs, values

def mean(x):
    return sum(x) / len(x)

def sampling_uncertainty(model, pathtoimg, unlabeled_data, add, device, selection_function,
                         quantile_min, quantile_max, type):
    if selection_function in ['min', 'max', 'mean']:
        if selection_function == 'min':
            fun = min
        elif selection_function == 'max':
            fun = max
        else:
            fun = mean
    else:
        fun = mean

    if type == 'faster':
        indexs, values, _ = find_out_net(model, device, pathtoimg, unlabeled_data, func=fun)
    else:
        indexs, values = find_out_net_custom(model, pathtoimg, unlabeled_data, func=fun)

    out_dict = {k: v for k, v in zip(indexs, values)}
    a = sorted(out_dict.items(), key=lambda x: x[1])
    pp = [(row[0], row[1]) for row in a if quantile_min <= row[1] < quantile_max]
    temp = random.sample(pp, k=min(add, len(pp)))
    if len(temp) < add:
        temp = temp + random.sample(list(set(a) - set(temp)), k=add - len(temp))

    # temp = a[-add:]
    out_name = [unlabeled_data[k] for k, v in temp]
    return sorted(out_name), values

def calc_faster(all_img, images, pathtoimg, pathtolabels,
                pathtoimgval, pathtolabelsval,
                add, device,
                path_model, batch_unlabeled, pretrain,
                save_model, use_val_test, retrain, selection_function,
                quantile_min, quantile_max, path_do_dir_model, num_epochs):
    if path_model == '':
        write_to_log('start train model')
        model0 = train_model(pathtoimg, pathtolabels, pathtoimgval, pathtolabelsval,
                             device, num_epochs=num_epochs, pretrain=pretrain, use_val_test=use_val_test)
    elif retrain:
        write_to_log('load and train model')
        if os.path.exists(path_model):
            premod = torch.load(path_model)
            model0 = train_model(pathtoimg, pathtolabels, pathtoimgval, pathtolabelsval,
                                 device, num_epochs=30, pretrain=pretrain, use_val_test=use_val_test,
                                 premodel=premod)
        else:
            return {'info': 'weight not exist'}

    else:
        write_to_log('load model')
        if os.path.exists(path_model):
            model0 = torch.load(path_model)
        else:
            return {'info': 'weight not exist'}
    unlabeled_data = list(set(all_img) - set([x[0] for x in images]))
    if batch_unlabeled > 0:
        unlabeled_data = random.sample(unlabeled_data, k=min(batch_unlabeled, len(unlabeled_data)))

    # methode = 'uncertainty'
    write_to_log('start uncertainty {}'.format(add))
    add_to_label_items, all_value = sampling_uncertainty(model0, pathtoimg, unlabeled_data, add, device, selection_function,
                                              quantile_min, quantile_max, 'faster')
    if save_model:
        path_model = os.path.join(path_do_dir_model, '{}.pth'.format(uuid.uuid4()))
        torch.save(model0, path_model)
        return {'data': add_to_label_items, 'model': path_model, 'all_value': all_value}
    else:
        return {'data': add_to_label_items, 'all_value': all_value}

def calc_custom(all_img, images, pathtoimg, pathtolabels,
                pathtoimgval, pathtolabelsval,
                add, device,
                path_model, batch_unlabeled, pretrain,
                save_model, use_val_test, retrain, selection_function,
                quantile_min, quantile_max, path_do_dir_model, num_epochs):
    dir_train_custom_model = ''
    if path_model == '':
        write_to_log('start train model')
        path_model, dir_train_custom_model = train_custom(pathtoimg, pathtolabels, pathtoimgval,
                                                          pathtolabelsval, pretrain, num_epochs)
    elif retrain:
        write_to_log('load and train model')
        if os.path.exists(path_model):
            path_model, dir_train_custom_model = train_custom(pathtoimg, pathtolabels, pathtoimgval,
                                                              pathtolabelsval, pretrain, num_epochs,
                                                              path_model)
        else:
            return {'info': 'weight not exist'}
    else:
        write_to_log('load model')
        if os.path.exists(path_model):
            pass
        else:
            return {'info': 'weight not exist'}
    unlabeled_data = list(set(all_img) - set([x[0] for x in images]))
    if batch_unlabeled > 0:
        unlabeled_data = random.sample(unlabeled_data, k=min(batch_unlabeled, len(unlabeled_data)))

    # methode = 'uncertainty'
    add_to_label_items = []
    write_to_log('start uncertainty {}'.format(add))
    add_to_label_items, all_value = sampling_uncertainty(path_model, pathtoimg, unlabeled_data, add, device, selection_function,
                                              quantile_min, quantile_max, 'custom')
    if save_model:
        return {'data': add_to_label_items, 'model': path_model, 'all_value': all_value}

    else:
        if len(dir_train_custom_model) > 0:
            shutil.rmtree(dir_train_custom_model)
        return {'data': add_to_label_items, 'all_value': all_value}

def train_api(path_to_img_train, path_to_labels_train,
              path_to_img_val, path_to_labels_val,
              add=100, gpu='0',
              path_model='', batch_unlabeled=-1, pretrain_from_hub=True,
              save_model=False, use_val_test_in_train=True, retrain_user_model=False, bbox_selection_policy='min',
              quantile_min=0, quantile_max=1, type_model='custom', num_epochs=10):
    device = f"cuda:{gpu}" if torch.cuda.is_available() else "cpu"
    path_do_dir_model = '/weight'
    write_to_log(device)

    all_img = os.listdir(path_to_img_train)
    images, _ = prepare_items_od(path_to_img_train, path_to_labels_train)
    if type_model == 'fasterrcnn':
        return calc_faster(all_img, images, path_to_img_train, path_to_labels_train,
                path_to_img_val, path_to_labels_val,
                add, device,
                path_model, batch_unlabeled, pretrain_from_hub,
                save_model, use_val_test_in_train, retrain_user_model, bbox_selection_policy,
                quantile_min, quantile_max, path_do_dir_model, num_epochs)
    elif type_model == 'custom':
        return calc_custom(all_img, images, path_to_img_train, path_to_labels_train,
                path_to_img_val, path_to_labels_val,
                add, device,
                path_model, batch_unlabeled, pretrain_from_hub,
                save_model, use_val_test_in_train, retrain_user_model, bbox_selection_policy,
                quantile_min, quantile_max, path_do_dir_model, num_epochs)
    else:
        return {'info': 'type_model is incorrect'}


def mAP(model0, pathtolabelsval, pathtoimgval, devicerest):
    images_test, annotations_test = prepare_items_od(pathtoimgval, pathtolabelsval)
    dataset_test = Dataset_objdetect(pathtoimgval, images_test, annotations_test, get_transform(), name='val')
    data_loader_test = DataLoader(dataset_test, batch_size=32, shuffle=False, collate_fn=utils.collate_fn)

    coco_evaluator = evaluate(model0, data_loader_test, device=devicerest)
    return {'mAP(0.5:0.95)': _summarize(coco_evaluator.coco_eval['bbox']), 'model': model0}


def _summarize(coco, ap=1, iouThr=None, areaRng='all', maxDets=100):
    p = coco.params
    iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
    titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
    typeStr = '(AP)' if ap == 1 else '(AR)'
    iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
        if iouThr is None else '{:0.2f}'.format(iouThr)

    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    if ap == 1:
        # dimension of precision: [TxRxKxAxM]
        s = coco.eval['precision']
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, :, aind, mind]
    else:
        # dimension of recall: [TxKxAxM]
        s = coco.eval['recall']
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        s = s[:, :, aind, mind]
    if len(s[s > -1]) == 0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s > -1])
    # print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
    return mean_s

def train_classification_and_predict(device, pathtoimg, images, path_to_classes, annotations, unlabeled_data):
    out = classification(device, pathtoimg, images, path_to_classes, annotations, unlabeled_data)
    return out

if __name__ == '__main__':
    path_to_img = '/home/neptun/PycharmProjects/datasets/coco/train2017'
    path_to_labels = '/home/neptun/PycharmProjects/datasets/coco/labelstrain'
    # path_to_boxes = '/home/neptun/PycharmProjects/datasets/coco/boxes/'

    # for i in os.listdir(path_to_boxes):
    #     os.remove(os.path.join(path_to_boxes, i))

    # c = train_api(path_to_img, path_to_labels, path_to_boxes, templates['num_for_al'], 'gpu')

    # print(c['data'])
