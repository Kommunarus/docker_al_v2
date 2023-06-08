import os
import random
import json
import copy
# from coco_lib.objectdetection import ObjectDetectionDataset
import shutil

# path_to_dataset = '/home/neptun/PycharmProjects/datasets/coco'
path_to_dataset = '/media/alex/DAtA4/Datasets/coco'
name_my_dataset = 'my_dataset_person_and_other'
def make_file(p):
    current_label = 1  #cat
    # N = 1000

    with open(os.path.join(path_to_dataset, 'instances_train2017.json')) as f:
        razmetka = json.load(f)

    categories = razmetka['categories']
    annotations = razmetka['annotations']
    images = razmetka['images']
    info = razmetka['info']
    licenses = razmetka['licenses']

    dict_w_h = {}
    for row in images:
        dict_w_h[row['id']] = row['height'] * row['width']

    all_photo = []
    for row in annotations:
        all_photo.append(row['image_id'])
    all_photo = list(set(all_photo))
    # N = len(all_photo)

    new_annotation = []
    for row in annotations:
        if row['category_id'] == current_label and \
                row['image_id'] in all_photo and \
                row['area'] / dict_w_h[row['image_id']] > 0.05:

            copy_row = copy.deepcopy(row)
            copy_row['segmentation'] = []
            new_annotation.append(copy_row)

    good_images_ids = []
    for row in new_annotation:
        good_images_ids.append(row['image_id'])
    good_images_ids = list(set(good_images_ids))
    bad_images_ids = list(set(all_photo) - set(good_images_ids))

    test_im = random.sample(good_images_ids, k=int(p/100 * len(good_images_ids)))
    train_im = list(set(good_images_ids) - set(test_im))

    # test
    good_images_path = []
    for row in images:
        if row['id'] in test_im:
            good_images_path.append(row['file_name'])
    good_images_path = list(set(good_images_path))

    for row in good_images_path:
        f1 = path_to_dataset + '/train2017/' + row
        f2 = path_to_dataset + f'/{name_my_dataset}/test/' + row
        shutil.copyfile(f1, f2)

    print('zero test file {} / {}'.format(len(test_im), len(good_images_path)))

    new_image = []

    for row in images:
        if row['id'] in test_im:
            copy_row = copy.deepcopy(row)
            new_image.append(copy_row)

    new_ann = []
    for row in annotations:
        if row['category_id'] == current_label and \
                row['image_id'] in test_im and \
                row['area'] / dict_w_h[row['image_id']] > 0.05:
            copy_row = copy.deepcopy(row)
            copy_row['segmentation'] = []
            new_ann.append(copy_row)

    new_razmetka = dict(annotations=new_ann, images=new_image,
                        categories=categories, info=info, licenses=licenses)

    with open(os.path.join(path_to_dataset, name_my_dataset, 'labels_test', 'test.json'), 'w') as f:
        f.write(json.dumps(new_razmetka))

    # trasn
    train_im = train_im + bad_images_ids
    good_images_path = []
    for row in images:
        if row['id'] in train_im:
            good_images_path.append(row['file_name'])
    good_images_path = list(set(good_images_path))

    for row in good_images_path:
        f1 = path_to_dataset + '/train2017/' + row
        f2 = path_to_dataset + f'/{name_my_dataset}/train/' + row
        shutil.copyfile(f1, f2)

    print('zero train file {} / {}'.format(len(train_im), len(good_images_path)))

    new_image = []

    for row in images:
        if row['id'] in train_im:
            copy_row = copy.deepcopy(row)
            new_image.append(copy_row)

    new_ann = []
    for row in annotations:
        if row['category_id'] == current_label and \
                row['image_id'] in train_im and \
                row['area'] / dict_w_h[row['image_id']] > 0.05:
            copy_row = copy.deepcopy(row)
            copy_row['segmentation'] = []
            new_ann.append(copy_row)
    new_razmetka = dict(annotations=new_ann, images=new_image,
                        categories=categories, info=info, licenses=licenses)

    with open(os.path.join(path_to_dataset, name_my_dataset, 'labels_train', 'train.json'), 'w') as f:
        f.write(json.dumps(new_razmetka))


if __name__ == '__main__':
    make_file(10)
    # dataset = ObjectDetectionDataset.load(os.path.join(path_to_dataset, 'labelsval', 'val.json'))
    # pass