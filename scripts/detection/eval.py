from scripts.detection.engine import evaluate, evaluate_custom
from scripts.detection.train import train_model
from scripts.detection.unit import Dataset_objdetect, prepare_items_od, prepare_items_od_with_wh
from scripts.detection.train import train_custom
from torch.utils.data import DataLoader
import scripts.detection.utils as utils
from torchvision import transforms as t
import torch
import numpy as np
import yaml
import os
import uuid
from scripts.detection.unit import write_to_log
import pathlib
import subprocess
import sys


def get_transform():
    transforms = [t.ToTensor()]
    return t.Compose(transforms)

def eval_faster(path_to_img_train, path_to_labels_train,
         path_to_img_val, path_to_labels_val,
         path_to_img_test, path_to_labels_test, device,
         save_model, pretrain, path_model, retrain, path_do_dir_model, num_epochs):
    if path_model == '':
        write_to_log('start train model')
        model0 = train_model(path_to_img_train, path_to_labels_train,
                             path_to_img_val, path_to_labels_val,
                             device,
                             num_epochs=num_epochs, pretrain=pretrain, use_val_test=True)
    elif retrain:
        write_to_log('load and train model')
        if os.path.exists(path_model):
            premod = torch.load(path_model)
            model0 = train_model(path_to_img_train, path_to_labels_train,
                                 path_to_img_val, path_to_labels_val,
                                 device, num_epochs=num_epochs, pretrain=pretrain, use_val_test=True,
                                 premodel=premod)
        else:
            return {'info': 'weight not exist'}

    else:
        write_to_log('load model')
        if os.path.exists(path_model):
            model0 = torch.load(path_model)
        else:
            return {'info': 'weight not exist'}

    images_test, annotations_test = prepare_items_od(path_to_img_test, path_to_labels_test)
    dataset_test = Dataset_objdetect(path_to_img_test, images_test, annotations_test, get_transform(), name='test',
                                     N=-1)
    data_loader_test = DataLoader(dataset_test, batch_size=8, shuffle=False, collate_fn=utils.collate_fn)
    write_to_log('in test {} samples'.format(len(set(images_test))))

    coco_evaluator = evaluate(model0, data_loader_test, device=device)
    dataset_test.f5.close()
    if save_model:
        path_model = os.path.join(path_do_dir_model, '{}.pth'.format(uuid.uuid4()))
        torch.save(model0, path_model)
        return {'mAP(0.5:0.95)': _summarize(coco_evaluator.coco_eval['bbox']),
                'model': path_model}
    else:
        return {'mAP(0.5:0.95)': _summarize(coco_evaluator.coco_eval['bbox'])}

def eval_custom(pathtoimg, pathtolabels, pathtoimgval, pathtolabelsval, path_to_img_test, path_to_labels_test,
                pretrain, path_model, retrain, device, save_model, num_epochs):
    if path_model == '':
        write_to_log('start train model')
        path_model = train_custom(pathtoimg, pathtolabels, pathtoimgval, pathtolabelsval, pretrain, num_epochs)
    elif retrain:
        write_to_log('load and train model')
        if os.path.exists(path_model):
            path_model = train_custom(pathtoimg, pathtolabels, pathtoimgval, pathtolabelsval, pretrain, num_epochs,
                                      path_model)
        else:
            return {'info': 'weight not exist'}
    else:
        write_to_log('load model')
        if os.path.exists(path_model):
            pass
        else:
            return {'info': 'weight not exist'}
    images_test, annotations_test = prepare_items_od(path_to_img_test, path_to_labels_test)
    dataset_test = Dataset_objdetect(path_to_labels_test, images_test, annotations_test, get_transform(), name='test',
                                     N=-1)
    data_loader_test = DataLoader(dataset_test, batch_size=8, shuffle=False, collate_fn=utils.collate_fn)
    write_to_log('in test {} samples'.format(len(set(images_test))))

    coco_evaluator = evaluate_custom(path_model, path_to_img_test, path_to_labels_test,
                                     data_loader_test, device=device)
    if save_model:
        return {'mAP(0.5:0.95)': _summarize(coco_evaluator.coco_eval['bbox']),
                'model': path_model}
    else:
        return {'mAP(0.5:0.95)': _summarize(coco_evaluator.coco_eval['bbox'])}

def eval(path_to_img_train, path_to_labels_train,
         path_to_img_val, path_to_labels_val,
         path_to_img_test, path_to_labels_test,
         device_rest, save_model, pretrain=True, path_model='', retrain=False, type_model='yolo', num_epochs=10):

    device = f"cuda:{device_rest}" if torch.cuda.is_available() else "cpu"
    path_do_dir_model = '/weight'
    write_to_log('eval')
    write_to_log(device)
    if type_model == 'fasterrcnn':
        return eval_faster(path_to_img_train, path_to_labels_train,
         path_to_img_val, path_to_labels_val,
         path_to_img_test, path_to_labels_test, device_rest,
         save_model, pretrain, path_model, retrain, path_do_dir_model, num_epochs)
    else:
        return eval_custom(path_to_img_train, path_to_labels_train,
                           path_to_img_val, path_to_labels_val,
                           path_to_img_test, path_to_labels_test,
                           pretrain, path_model, retrain, device, save_model, num_epochs)


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
    print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
    return mean_s



if __name__ == '__main__':
    path_to_img_train = '/home/neptun/PycharmProjects/datasets/coco/my_dataset/train'
    path_to_labels_train = '/home/neptun/PycharmProjects/datasets/coco/labelstrain/first.json'
    path_to_img_val = '/home/neptun/PycharmProjects/datasets/coco/my_dataset/val'
    path_to_labels_val = '/home/neptun/PycharmProjects/datasets/coco/my_dataset/labels_val/val.json'
    path_to_img_test = '/home/neptun/PycharmProjects/datasets/coco/my_dataset/test'
    path_to_labels_test = '/home/neptun/PycharmProjects/datasets/coco/my_dataset/labels_test/test.json'
    gpu = 0
    pretrain_from_hub = False
    save_model = False
    path_model = ''
    retrain_user_model = False
    type_model = 'fasterrcnn'
    num_epochs = 2

    coco_evaluator = eval(path_to_img_train, path_to_labels_train,
         path_to_img_val, path_to_labels_val,
         path_to_labels_test, path_to_img_test,
         gpu, save_model, pretrain=pretrain_from_hub, path_model=path_model, retrain=retrain_user_model,
                          type_model=type_model,
         num_epochs=num_epochs)

    print(coco_evaluator)