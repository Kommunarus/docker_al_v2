from scripts.detection.engine import evaluate
from scripts.detection.train import train_model
from scripts.detection.unit import Dataset_objdetect, prepare_items_od
from torch.utils.data import DataLoader
import scripts.detection.utils as utils
from torchvision import transforms as t
import torch
import numpy as np
import yaml
# with open('../detection/setting.yaml') as f:
# with open('setting.yaml') as f:
with open('scripts/detection/setting.yaml') as f:
    templates = yaml.safe_load(f)

def get_transform():
    transforms = [t.ToTensor()]
    return t.Compose(transforms)


def eval(path_to_labels_train, path_to_img_train, path_to_labels_val, path_to_img_val, device_rest, model=None):

    if device_rest == 'gpu':
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    if model is None:
        images_train, annotations_train = prepare_items_od(path_to_img_train, path_to_labels_train)
        model0 = train_model(path_to_img_train, path_to_labels_train, path_to_img_val, path_to_labels_val,
                             images_train, annotations_train, device,
                             num_epochs=templates['n_epoch'])
    else:
        model0 = model

    images_test, annotations_test = prepare_items_od(path_to_img_val, path_to_labels_val)
    dataset_test = Dataset_objdetect(path_to_img_val, images_test, annotations_test, get_transform())
    data_loader_test = DataLoader(dataset_test, batch_size=32, shuffle=False, collate_fn=utils.collate_fn)

    coco_evaluator = evaluate(model0, data_loader_test, device=device)
    # return {'mAP(0.5:0.95)': _summarize(coco_evaluator.coco_eval['bbox'])}
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