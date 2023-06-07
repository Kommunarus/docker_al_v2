import os
import random
import json
import copy


def make_file(N, path_to_json_train, path_to_out):
    current_label = 1  #cat
    # N = 1000
    # path_to_dataset = '/media/alex/DAtA2/Datasets/coco'
    # path_to_dataset = '/home/neptun/PycharmProjects/datasets/coco'

    with open(path_to_json_train) as f:
        razmetka = json.load(f)

    categories = razmetka['categories']
    annotations = razmetka['annotations']
    images = razmetka['images']
    info = razmetka['info']
    licenses = razmetka['licenses']

    all_photo = []
    for row in annotations:
        all_photo.append(row['image_id'])
    all_photo = list(set(all_photo))

    if N != -1:
        all_photo = random.sample(all_photo, k=N)

    new_annotation = []
    for row in annotations:
        if row['category_id'] == current_label and \
                row['image_id'] in all_photo:
                # row['area'] > 10\

            copy_row = copy.deepcopy(row)
            copy_row['segmentation'] = []
            new_annotation.append(copy_row)

    new_image = []

    for row in images:
        if row['id'] in all_photo:
            copy_row = copy.deepcopy(row)
            new_image.append(copy_row)


    new_razmetka = dict(annotations=new_annotation, images=new_image,
                        categories=categories, info=info, licenses=licenses)

    with open(path_to_out, 'w') as f:
        f.write(json.dumps(new_razmetka))

if __name__ == '__main__':
    make_file(1000,
              path_to_json_train='/home/neptun/PycharmProjects/datasets/coco/my_dataset/labels_train/train.json',
              path_to_out='/home/neptun/PycharmProjects/datasets/coco/first.json')