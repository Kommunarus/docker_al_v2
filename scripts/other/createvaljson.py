import os
import random
import json
import copy
# from coco_lib.objectdetection import ObjectDetectionDataset
import shutil

# path_to_dataset = '/home/neptun/PycharmProjects/datasets/coco'
path_to_dataset = '/media/alex/DAtA4/Datasets/coco'
name_my_dataset = 'my_dataset_person_and_other'

def make_file(N):
    current_label = 1  #cat
    # N = 1000

    with open(os.path.join(path_to_dataset, 'instances_val2017.json')) as f:
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

    if N != -1:
        all_photo = random.sample(all_photo, k=N)

    new_annotation = []
    for row in annotations:
        iii = [x for x in images if x['id'] == row['image_id']][0]
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


    good_images_path = []
    for row in images:
        if row['id'] in good_images_ids:
            good_images_path.append(row['file_name'])
    good_images_path = list(set(good_images_path))

    for row in good_images_path:
        f1 = path_to_dataset + '/val2017/' + row
        f2 = path_to_dataset + f'/{name_my_dataset}/val/' + row
        shutil.copyfile(f1, f2)

    print('zero file {} / {}'.format(len(good_images_ids), N))

    new_image = []


    for row in images:
        if row['id'] in good_images_ids:
            copy_row = copy.deepcopy(row)
            new_image.append(copy_row)


    new_razmetka = dict(annotations=new_annotation, images=new_image,
                        categories=categories, info=info, licenses=licenses)

    with open(os.path.join(path_to_dataset, name_my_dataset, 'labels_val', 'val.json'), 'w') as f:
        f.write(json.dumps(new_razmetka))

if __name__ == '__main__':
    make_file(-1)
    # dataset = ObjectDetectionDataset.load(os.path.join(path_to_dataset, 'labelsval', 'val.json'))
    # pass