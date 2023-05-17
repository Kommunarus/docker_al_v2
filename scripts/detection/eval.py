from scripts.detection.engine import evaluate
from scripts.detection.train import train_model
from scripts.detection.unit import Dataset_objdetect, prepare_items_od
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


def get_transform():
    transforms = [t.ToTensor()]
    return t.Compose(transforms)


def eval(path_to_img_train, path_to_labels_train,
         path_to_img_val, path_to_labels_val,
         path_to_labels_test, path_to_img_test,
         device_rest, save_model, pretrain=True, path_model='', retrain=False):
    write_to_log('param:')
    write_to_log('path_to_img_train: {}'.format(path_to_img_train))
    list_file = os.listdir(path_to_img_train)
    write_to_log('count train files: {}'.format(len(list_file)))

    write_to_log('path_to_labels_train: {}'.format(path_to_labels_train))
    p = pathlib.Path(path_to_labels_train)
    stat_info = p.stat()
    write_to_log('size train annotation: {}'.format(stat_info.st_size))

    write_to_log('path_to_img_val: {}'.format(path_to_img_val))
    list_file = os.listdir(path_to_img_val)
    write_to_log('count val files: {}'.format(len(list_file)))

    write_to_log('path_to_labels_val: {}'.format(path_to_labels_val))
    p = pathlib.Path(path_to_labels_val)
    stat_info = p.stat()
    write_to_log('size val annotation: {}'.format(stat_info.st_size))

    write_to_log('path_to_labels_test: {}'.format(path_to_labels_test))
    list_file = os.listdir(path_to_img_val)
    write_to_log('count test files: {}'.format(len(list_file)))

    write_to_log('path_to_img_test: {}'.format(path_to_img_test))
    p = pathlib.Path(path_to_img_test)
    stat_info = p.stat()
    write_to_log('size val annotation: {}'.format(stat_info.st_size))

    write_to_log('device_rest: {}'.format(device_rest))
    write_to_log('save_model: {}'.format(save_model))
    write_to_log('pretrain: {}'.format(pretrain))
    write_to_log('path_model: {}'.format(path_model))
    write_to_log('retrain: {}'.format(retrain))


    device = f"cuda:{device_rest}" if torch.cuda.is_available() else "cpu"
    path_do_dir_model = '/weight'
    write_to_log('eval')
    write_to_log(device)

    if path_model == '':
        write_to_log('start train model')
        model0 = train_model(path_to_img_train, path_to_labels_train,
                             path_to_img_val, path_to_labels_val,
                             device,
                             num_epochs=30, pretrain=pretrain, use_val_test=True)
    elif retrain:
        write_to_log('load and train model')
        if os.path.exists(path_model):
            premod = torch.load(path_model)
            model0 = train_model(path_to_img_train, path_to_labels_train,
                                 path_to_img_val, path_to_labels_val,
                                 device, num_epochs=30, pretrain=pretrain, use_val_test=True,
                                 premodel=premod)
        else:
            return {'info': 'weight not exist'}

    else:
        write_to_log('load model')
        if os.path.exists(path_model):
            model0 = torch.load(path_model)
        else:
            return {'info': 'weight not exist'}


    images_test, annotations_test = prepare_items_od(path_to_labels_test, path_to_img_test)
    dataset_test = Dataset_objdetect(path_to_labels_test, images_test, annotations_test, get_transform(), name='test')
    data_loader_test = DataLoader(dataset_test, batch_size=32, shuffle=False, collate_fn=utils.collate_fn)
    write_to_log('in test {} samples'.format(len(set(images_test))))


    coco_evaluator = evaluate(model0, data_loader_test, device=device)
    if save_model:
        path_model = os.path.join(path_do_dir_model, '{}.pth'.format(uuid.uuid4()))
        torch.save(model0, path_model)
        return {'mAP(0.5:0.95)': _summarize(coco_evaluator.coco_eval['bbox']),
                'model': path_model}
    else:
        return {'mAP(0.5:0.95)': _summarize(coco_evaluator.coco_eval['bbox'])}
    # return {'mAP(0.5:0.95)': _summarize(coco_evaluator.coco_eval['bbox'])}


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
    path_to_img_val = '/home/neptun/PycharmProjects/datasets/coco/val2017/'
    path_to_labels_val = '/home/neptun/PycharmProjects/datasets/coco/labelsval/'
    path_to_img_train = '/home/neptun/PycharmProjects/datasets/coco/train2017/'
    path_to_labels_train = '/home/neptun/PycharmProjects/datasets/coco/labelstrain/'
    device_rest = 'gpu'

    coco_evaluator = eval(path_to_labels_train, path_to_img_train, path_to_labels_val, path_to_img_val, device_rest)

    print(coco_evaluator)