import math
import sys
import time
import os
import uuid
import subprocess
import shutil
import albumentations

import torch
import torchvision.models.detection.mask_rcnn
import scripts.detection.utils as utils
from scripts.detection.coco_eval import CocoEvaluator
from scripts.detection.coco_utils import get_coco_api_from_dataset
from scripts.detection.unit import prepare_items_od_with_wh


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets, _, _ in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image r_u_n on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets, _, _ in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    # coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

@torch.inference_mode()
def evaluate_custom(path_model, path_to_img_test, path_to_labels_test,
                    data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image r_u_n on the GPU
    images_test, _ = prepare_items_od_with_wh(path_to_img_test, path_to_labels_test)
    script = os.getcwd() + '/custom_model/detect.py'

    data_file = os.getcwd() + f'/data/{"test_" + str(uuid.uuid4())}.txt'
    with open(data_file, 'w') as list_file:
        for _, _, _, name_files in data_loader:
            for f in name_files:
                list_file.write('{}/{}\n'.format(path_to_img_test, f.decode("utf-8")))

    path_to_out = os.getcwd() + f'/data/{str(uuid.uuid4())}'
    os.makedirs(path_to_out)

    subprocess.check_call([sys.executable, script, "--source", data_file, "--weights", path_model, "--img", '640',
                           '--project', path_to_out, '--save-txt', '--agnostic-nms', '--nosave', '--save-conf'])


    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ["bbox"]
    coco_evaluator = CocoEvaluator(coco, iou_types)
    wh = {}
    for line in images_test:
        wh[line[0]] = (line[2], line[3])

    for _, targets, _, name_file in metric_logger.log_every(data_loader, 100, header):
        model_time = time.time()
        boxes = []
        scores = []
        labels = []
        for b in name_file:
            name_txt = os.path.join(path_to_out, 'exp', 'labels', b.decode("utf-8").split('.')[0] + '.txt')
            boxes_p = []
            scores_p = []
            labels_p = []

            w_image = wh[b.decode("utf-8")][0]
            h_image = wh[b.decode("utf-8")][1]
            if w_image > h_image:
                k_x = 1
                k_y = w_image / h_image
            elif h_image > w_image:
                k_x = h_image / w_image
                k_y = 1
            else:
                k_x = 1
                k_y = 1

            if os.path.exists(name_txt):
                with open(name_txt) as f_t:
                    for line in f_t.readlines():
                        arr = line.strip().split(' ')
                        labels_p.append(1)
                        xc = float(arr[1]) * w_image
                        yc = float(arr[2]) * h_image
                        w = float(arr[3]) * w_image
                        h = float(arr[4]) * h_image
                        x1 = max(xc - w/2, 0) * k_x
                        y1 = max(yc - h/2, 0) * k_y
                        x2 = min(xc + w/2, w_image) * k_x
                        y2 = min(yc + h/2, h_image) * k_y

                        boxes_p.append([x1, y1, x2, y2])
                        scores_p.append(float(arr[5]))
            if len(boxes_p) == 0:
                labels_p.append(1)
                boxes_p.append([0, 0, 1, 1])
                scores_p.append(0.1)
            boxes.append(boxes_p)
            scores.append(scores_p)
            labels.append(labels_p)



        # outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        outputs = [{'boxes': torch.Tensor(x), 'scores': torch.Tensor(y), 'labels': torch.Tensor(z).to(torch.int64)}
                   for x, y, z in zip(boxes, scores, labels)]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # del my files
    os.remove(data_file)
    shutil.rmtree(path_to_out)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    # coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
