from scripts.detection.unit import prepare_items_od_with_wh
import uuid
import os
import shutil
import copy

def json_to_yolo(pathtoimg, pathtolabels, pathtoimgval, pathtolabelsval):
    # path_root = '/home/neptun/PycharmProjects/docker_al_v2/data'
    path_root = 'data/'
    images_train, annotations_train = prepare_items_od_with_wh(pathtoimg, pathtolabels)
    images_val, annotations_val = prepare_items_od_with_wh(pathtoimgval, pathtolabelsval)
    name_dir = str(uuid.uuid4())
    os.makedirs(f'{path_root}/{name_dir}/train/images')
    os.makedirs(f'{path_root}/{name_dir}/train/labels')
    os.makedirs(f'{path_root}/{name_dir}/val/images')
    os.makedirs(f'{path_root}/{name_dir}/val/labels')
    dict_id_train = {}
    dict_id_val = {}

    for name, id, w, h in images_train:
        src = os.path.join(pathtoimg, name)
        dst = os.path.join(path_root, name_dir, 'train', 'images', name)
        shutil.copyfile(src, dst)
        dict_id_train[id] = (name, w, h)

    for name, id, w, h in images_val:
        src = os.path.join(pathtoimgval, name)
        dst = os.path.join(path_root, name_dir, 'val', 'images', name)
        shutil.copyfile(src, dst)
        dict_id_val[id] = (name, w, h)

    dict_an_train = {}
    dict_an_val = {}
    for box, id in annotations_train:
        value = dict_an_train.get(id)
        if value is None:
            dict_an_train[id] = [box,]
        else:
            copy_val = copy.deepcopy(value)
            copy_val.append(box)
            dict_an_train[id] = copy_val
    for k, v in dict_an_train.items():
        lab_name = dict_id_train[k][0].split('.')[0]
        w_im = dict_id_train[k][1]
        h_im = dict_id_train[k][2]
        with open(f'{path_root}/{name_dir}/train/labels/{lab_name}.txt', 'w') as f:
            for line in v:
                x, y, w, h = line
                xc = x + w / 2
                yc = y + h / 2
                f.write('0 {} {} {} {}\n'.format(xc/w_im, yc/h_im, w/w_im, h/h_im))

    for box, id in annotations_val:
        value = dict_an_val.get(id)
        if value is None:
            dict_an_val[id] = [box,]
        else:
            copy_val = copy.deepcopy(value)
            copy_val.append(box)
            dict_an_val[id] = copy_val
    for k, v in dict_an_val.items():
        lab_name = dict_id_val[k][0].split('.')[0]
        w_im = dict_id_val[k][1]
        h_im = dict_id_val[k][2]
        with open(f'{path_root}/{name_dir}/val/labels/{lab_name}.txt', 'w') as f:
            for line in v:
                x, y, w, h = line
                xc = x + w / 2
                yc = y + h / 2
                f.write('0 {} {} {} {}\n'.format(xc/w_im, yc/h_im, w/w_im, h/h_im))

    return path_root, name_dir

if __name__ == '__main__':
    ds = '/home/neptun/PycharmProjects/datasets'
    pathtoimg = ds + '/coco/my_dataset/train'
    pathtolabels = ds + '/coco/labelstrain/first.json'
    pathtoimgval = ds + '/coco/my_dataset/val'
    pathtolabelsval = ds + '/coco/my_dataset/labels_val/val.json'
    json_to_yolo(pathtoimg, pathtolabels, pathtoimgval, pathtolabelsval)