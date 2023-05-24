import random
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
import torch
import json
import yaml
import os
from skimage import io
import numpy as np
import h5py
import uuid
import datetime


mean = [0.4914, 0.4822, 0.4465]
std = [0.2470, 0.2435, 0.2616]

data_transform = transforms.Compose([
    transforms.ToPILImage(),
    # transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean,  std=std)
])


# Object detection
def prepare_items_od_with_wh(path_to_img, path_to_labels):

    if os.path.isfile(path_to_labels):
        with open(path_to_labels) as f:
            razmetka = json.load(f)

        annotations = razmetka['annotations']
        images = razmetka['images']
    else:
        annotations = []
        images = []
        for file in os.listdir(path_to_labels):
            with open(os.path.join(path_to_labels, file)) as f:
                razmetka = json.load(f)

            annotations = annotations + razmetka['annotations']
            images = images + razmetka['images']
    images_short = [(row['file_name'], row['id'], row['width'], row['height']) for row in images]
    annotations_short = [(row['bbox'], row['image_id']) for row in annotations]
    images_short = list(set(images_short))
    return images_short, annotations_short

def prepare_items_od(path_to_img, path_to_labels):

    if os.path.isfile(path_to_labels):
        with open(path_to_labels) as f:
            razmetka = json.load(f)

        annotations = razmetka['annotations']
        images = razmetka['images']
    else:
        annotations = []
        images = []
        for file in os.listdir(path_to_labels):
            with open(os.path.join(path_to_labels, file)) as f:
                razmetka = json.load(f)

            annotations = annotations + razmetka['annotations']
            images = images + razmetka['images']
    images_short = [(row['file_name'], row['id']) for row in images]
    # images_dict = {row[0]: row[1] for row in images_short}

    annotations_short = [(row['bbox'], row['image_id']) for row in annotations]
    images_short = list(set(images_short))
    return images_short, annotations_short


class Dataset_objdetect(Dataset):
    def __init__(self, path_to_img, images, annotations, transforms, path_to_h5=os.getcwd()+'/data', name=''):
        self.path_to_img = path_to_img
        count = 0
        if not annotations is None:
            new_id = []
            for row in annotations:
                new_id.append(row[1])
            new_images = []
            for row in images:
                if row[1] in new_id:
                    new_images.append(row)
            self.images = new_images

            self.transA = A.Compose([A.Resize(640, 640)], bbox_params=A.BboxParams(format='coco',
                                                                                   label_fields=['class_labels']))

            # create h5
            if name != '':
                nm = os.path.join(path_to_h5, "{}.hdf5".format(name))
                if not os.path.exists(nm):
                    create_h5 = True
                    f = h5py.File(nm, "w")
                else:
                    create_h5 = False
                    f = h5py.File(nm, "r")
                self.name = name
            else:
                id_ds = str(uuid.uuid4())
                f = h5py.File(os.path.join(path_to_h5, "{}.hdf5".format(id_ds)), "w")
                create_h5 = True
                self.name = id_ds
            self.create_dataset = create_h5


            if create_h5:
                pad = 100

                list_imgs = []
                list_boxs = []
                list_labels = []
                list_area = []
                list_crowed = []
                list_num = []
                for img in new_images:
                    name_file = img[0]
                    id_file = img[1]
                    path = os.path.join(self.path_to_img, name_file)

                    image = io.imread(path)
                    if len(image.shape) != 3:
                        continue
                    boxs = [x[0] for x in annotations if x[1] == id_file]

                    num_objs = len(boxs)
                    class_labels = ['tag'] * num_objs
                    transformed = self.transA(image=image, bboxes=boxs, class_labels=class_labels)
                    transformed_image = transformed['image']
                    transformed_bboxes = transformed['bboxes']

                    transformed_bboxes = np.array([[x[0], x[1], x[0]+x[2], x[1]+x[3]] for x in transformed_bboxes])
                    transformed_bboxes.resize((pad, 4), refcheck=False)

                    labels = np.ones((num_objs,), dtype=np.uint8)
                    labels.resize((pad, 1), refcheck=False)


                    area = (transformed_bboxes[:, 3] - transformed_bboxes[:, 1]) * \
                           (transformed_bboxes[:, 2] - transformed_bboxes[:, 0])

                    iscrowd = np.zeros((num_objs,), dtype=np.uint8)
                    iscrowd.resize((pad, 1), refcheck=False)


                    list_imgs.append(transformed_image)
                    list_boxs.append(transformed_bboxes)
                    list_labels.append(labels)
                    list_area.append(area)
                    list_crowed.append(iscrowd)
                    list_num.append(num_objs)

                    count += 1

                images = np.array(list_imgs)
                boxes = np.array(list_boxs)
                labels = np.array(list_labels)
                areas = np.array(list_area)
                crowed = np.array(list_crowed)
                numer = np.array(list_num)
                N= len(list_imgs)
                f.create_dataset('images', data=images, shape=(N, 640, 640, 3))
                f.create_dataset('boxes', data=boxes, shape=(N, pad, 4))
                f.create_dataset('labels', data=labels, shape=(N, pad, 1))
                f.create_dataset('areas', data=areas, shape=(N, pad, 1))
                f.create_dataset('crowed', data=crowed, shape=(N, pad, 1))
                f.create_dataset('num', data=numer, shape=(N, 1))

            self.f5 = f
        else:
            self.transB = A.Compose([A.Resize(640, 640)])
            self.images = images
            # create h5
            id_ds = str(uuid.uuid4())
            f = h5py.File(os.path.join(path_to_h5, "{}.hdf5".format(id_ds)), "w")
            list_imgs = []
            self.name = id_ds

            for img in images:
                path = os.path.join(self.path_to_img, img)
                image = io.imread(path)
                if len(image.shape) != 3:
                    continue

                transformed = self.transB(image=image)
                transformed_image = transformed['image']
                list_imgs.append(transformed_image)
                count += 1


            N= len(list_imgs)
            images = np.array(list_imgs)
            f.create_dataset('images', data=images, shape=(N, 640, 640, 3))
            self.create_dataset = True

            self.f5 = f

        self.count = len(self.f5.get('images')[:])

        self.annotations = annotations
        self.transforms = transforms

    def __getitem__(self, idx):
        # шаг обучения
        if not self.annotations is None:
            images = self.f5.get('images')[idx, :]
            boxes = self.f5.get('boxes')[idx, :]
            labels = self.f5.get('labels')[idx, :]
            area = self.f5.get('areas')[idx, :]
            iscrowd = self.f5.get('crowed')[idx, :]
            num = self.f5.get('num')[idx, 0]

            target = {}
            target["boxes"] = torch.tensor(boxes[:num])
            target["labels"] = torch.tensor(labels[:num].flatten()).to(torch.int64)
            target["image_id"] = torch.as_tensor(idx, dtype=torch.int64)
            target["area"] = torch.tensor(area[:num].flatten())
            target["iscrowd"] = torch.tensor(iscrowd[:num].flatten())

            if self.transforms is not None:
                images = self.transforms(images)

            return images, target, idx
        else:
            images = self.f5.get('images')[idx, :]
            if self.transforms is not None:
                images = self.transforms(images)
            return images, None, idx

    def __len__(self):
        return self.count

def write_to_log(txt):
    path = os.getcwd() + '/history.txt'
    with open(path, 'a') as file:
        file.write('{}\t{}\n'.format(datetime.datetime.now(), txt))

if __name__ == '__main__':
    pass