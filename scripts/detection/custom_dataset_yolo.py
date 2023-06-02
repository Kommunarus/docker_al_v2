from scripts.detection.unit import prepare_items_od_with_wh
import uuid
import os
import shutil
import copy
import subprocess
import sys

def json_to_yolo(pathtoimg, pathtolabels, pathtoimgval, pathtolabelsval):
    path_root = '/home/alex/PycharmProjects/docker_al_v2/data'
    # path_root = 'data/'
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

def json_one_to_yolo(pathtoimg, pathtolabels):
    path_root = '/home/alex/PycharmProjects/docker_al_v2/data'
    # path_root = 'data/'
    images_val, annotations_val = prepare_items_od_with_wh(pathtoimg, pathtolabels)
    name_dir = str(uuid.uuid4())
    os.makedirs(f'{path_root}/{name_dir}/val/images')
    os.makedirs(f'{path_root}/{name_dir}/val/labels')
    dict_id_val = {}

    for name, id, w, h in images_val:
        src = os.path.join(pathtoimg, name)
        dst = os.path.join(path_root, name_dir, 'val', 'images', name)
        shutil.copyfile(src, dst)
        dict_id_val[id] = (name, w, h)

    dict_an_val = {}
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

def calc_map50(path_to_img_test, path_to_labels_test, path_model, gpu):
    path_root, name_dir = json_one_to_yolo(path_to_img_test, path_to_labels_test)

    getcwd = '/home/alex/PycharmProjects/docker_al_v2'
    # script = os.path.join(getcwd, 'custom_model/val.py')
    path_to_data_dir = os.path.join(getcwd, f'data/{name_dir}')
    path_to_yaml = os.path.join(path_to_data_dir, 'my.yaml')
    text_yaml = '\n' + \
                f'path: {path_root + "/" + name_dir}\n' + \
                'train: \n' + \
                'val: val/images\n' + \
                'test:\n' + \
                '# Classes\n' + \
                'names:\n' + \
                '   0: obj\n'
    with open(path_to_yaml, 'w') as f:
        f.write(text_yaml)

    # out = subprocess.check_call([sys.executable, script, "--data", path_to_yaml, "--weights", path_model,
    #                        "--img", '640', "--project", path_to_data_dir,
    # ])
    import custom_model.val as validate
    results, _, _ = validate.run(data=path_to_yaml, weights=path_model, project=path_to_data_dir, device=gpu)
    shutil.rmtree(path_to_data_dir)
    return results

def save_images_detect(live, model, list_images, step, gpu):
    getcwd = '/home/alex/PycharmProjects/docker_al_v2'
    script = getcwd + '/custom_model/detect.py'

    data_file = getcwd + f'/data/{str(uuid.uuid4())}.txt'
    with open(data_file, 'w') as list_file:
        for f in list_images:
            list_file.write('{}\n'.format(f))

    path_to_out = getcwd + f'/data/{str(uuid.uuid4())}'
    os.makedirs(path_to_out)

    subprocess.check_call([sys.executable, script, "--source", data_file, "--weights", model, "--img", '640',
                           '--project', path_to_out, '--agnostic-nms',
                           '--conf-thres', '0.15', '--device', str(gpu)])

    for indx, file_img in enumerate(os.listdir(path_to_out+'/exp')):
        live.log_image(f"detect_{step}_{indx}.png", os.path.join(path_to_out, 'exp', file_img))
    shutil.rmtree(path_to_out)
    os.remove(data_file)

if __name__ == '__main__':
    ds = '/home/neptun/PycharmProjects/datasets'
    pathtoimg = ds + '/coco/my_dataset/train'
    pathtolabels = ds + '/coco/labelstrain/first.json'
    pathtoimgval = ds + '/coco/my_dataset/val'
    pathtolabelsval = ds + '/coco/my_dataset/labels_val/val.json'
    json_to_yolo(pathtoimg, pathtolabels, pathtoimgval, pathtolabelsval)