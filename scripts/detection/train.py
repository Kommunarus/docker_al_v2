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
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import GeneralizedRCNNTransform
from PIL import Image
import re
import uuid
from scripts.detection.unit import write_to_log


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
        data_loader_test = DataLoader(dataset_test, batch_size=32, shuffle=False, collate_fn=utils.collate_fn)


    num_classes = 2
    best_model = None
    best_mape = 0

    if premodel is None:
        model = get_model_instance_segmentation(num_classes, pretrain)
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
    if ds0.create_dataset:
        os.remove('../../data/'+ds0.name+'.hdf5')
    return best_model


def get_model_instance_segmentation(num_classes, pretrain):
    # load an instance segmentation model pre-trained on COCO
    if pretrain:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)


    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.transform = GeneralizedRCNNTransform(min_size=224, max_size=224, image_mean=[0.485, 0.456, 0.406],
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

def mean(x):
    return sum(x) / len(x)

def sampling_uncertainty(model, pathtoimg, unlabeled_data, add, device, selection_function,
                         quantile_min, quantile_max):
    # min, max, mean
    # func = mean
    if selection_function in ['min', 'max', 'mean']:
        if selection_function == 'min':
            fun = min
        elif selection_function == 'max':
            fun = max
        else:
            fun = mean
    else:
        fun = mean

    indexs, values, _ = find_out_net(model, device, pathtoimg, unlabeled_data, func=fun)

    # out_name = []
    out_dict = {k: v for k, v in zip(indexs, values)}
    a = sorted(out_dict.items(), key=lambda x: x[1])

    # temp = []
    # q_min = np.quantile(np.array(values), quantile_min)
    # q_max = np.quantile(np.array(values), quantile_max)

    pp = [(row[0], row[1]) for row in a if quantile_min <= row[1] < quantile_max]
    temp = random.sample(pp, k=min(add, len(pp)))
    # for ii in range(5):
    #     p_min = 0.5 + ii * 0.1
    #     p_max = 0.5 + (ii + 1) * 0.1
    #     pp = [(row[0], row[1]) for row in a if p_min <= row[1] < p_max]
    #     temp = temp + random.sample(pp, k=min(add//5, len(pp)))
    if len(temp) < add:
        temp = temp + random.sample(list(set(a) - set(temp)), k=add - len(temp))

    # temp = a[-add:]
    out_name = [unlabeled_data[k] for k, v in temp]
    return sorted(out_name)


def train_api(pathtoimg, pathtolabels,
              pathtoimgval, pathtolabelsval,
              add=100, device_rest='0',
              path_model='', batch_unlabeled=-1, pretrain=True,
              save_model=False, use_val_test=True, retrain=False, selection_function='min',
              quantile_min=0, quantile_max=1):
    device = f"cuda:{device_rest}" if torch.cuda.is_available() else "cpu"
    path_do_dir_model = '/weight'
    write_to_log(device)

    # print('curr dir{}'.format(os.path.curdir))
    all_img = os.listdir(pathtoimg)
    images, _ = prepare_items_od(pathtoimg, pathtolabels)
    if path_model == '':
        write_to_log('start train model')
        model0 = train_model(pathtoimg, pathtolabels, pathtoimgval, pathtolabelsval,
                             device, num_epochs=30, pretrain=pretrain, use_val_test=use_val_test)
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
    add_to_label_items = sampling_uncertainty(model0, pathtoimg, unlabeled_data, add, device, selection_function,
                                              quantile_min, quantile_max)
    if save_model:
        path_model = os.path.join(path_do_dir_model, '{}.pth'.format(uuid.uuid4()))
        torch.save(model0, path_model)
        return {'data': add_to_label_items, 'model': path_model}
    else:
        return {'data': add_to_label_items}

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
