import json
import os
import random
import argparse
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--split-type', default=1, help='use stratified group kfold: 1 or random split: 2')
parser.add_argument('-i', '--input-path', default='/opt/ml/dataset/train.json', help='input train json path')
parser.add_argument('-o', '--output-path', default='/opt/ml/dataset/kfold/', help='output dir path')
parser.add_argument('-v', '--val-ratio', default=0.2, help='validation split ratio')
parser.add_argument('-s', '--seed', default=41, help='random seed')
args = parser.parse_args()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def random_split_dataset(input_json, output_dir, val_ratio, seed):
    output_dir = output_dir.split('/')
    output_dir[-2] = 'randomsplit'
    output_dir = '/'.join(output_dir)
    print(output_dir)
    
    with open(input_json) as json_reader:
        dataset = json.load(json_reader)

    images = dataset['images']
    annotations = dataset['annotations']
    categories = dataset['categories']

    image_ids = [x.get('id') for x in images]
    image_ids.sort()
    random.shuffle(image_ids)

    num_val = int(len(image_ids) * val_ratio)
    num_train = len(image_ids) - num_val

    image_ids_val, image_ids_train = set(image_ids[:num_val]), set(image_ids[num_val:])

    train_images = [x for x in images if x.get('id') in image_ids_train]
    val_images = [x for x in images if x.get('id') in image_ids_val]
    train_annotations = [x for x in annotations if x.get('image_id') in image_ids_train]
    val_annotations = [x for x in annotations if x.get('image_id') in image_ids_val]

    train_data = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': categories,
    }

    val_data = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': categories,
    }

    output_seed_dir = os.path.join(output_dir, f'seed{seed}')
    os.makedirs(output_seed_dir, exist_ok=True)
    output_train_json = os.path.join(output_seed_dir, 'train.json')
    output_val_json = os.path.join(output_seed_dir, 'val.json')

    with open(output_train_json, 'w') as train_writer:
        json.dump(train_data, train_writer)
    print(f'done. {output_train_json}')

    
    with open(output_val_json, 'w') as val_writer:
        json.dump(val_data, val_writer)
    print(f'done. {output_val_json}')

def stratified_group_kfold_dataset(input_json, output_dir, seed):
    with open(input_json) as f:
        data = json.load(f)

    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    annotation_infos = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]
    X = np.ones((len(data['annotations']), 1))
    y = np.array([annotation_info[1] for annotation_info in annotation_infos])
    groups = np.array([annotation_info[0] for annotation_info in annotation_infos])

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    for idx, (train_ids, val_ids) in enumerate(cv.split(X, y, groups)):
        train_images = [x for x in images if x.get('id') in groups[train_ids]]
        val_images = [x for x in images if x.get('id') in groups[val_ids]]
        train_annotations = [x for x in annotations if x.get('image_id') in groups[train_ids]]
        val_annotations = [x for x in annotations if x.get('image_id') in groups[val_ids]]

        train_data = {
            'images': train_images,
            'annotations': train_annotations,
            'categories': categories,
        }

        val_data = {
            'images': val_images,
            'annotations': val_annotations,
            'categories': categories,
        }
        
        output_seed_dir = os.path.join(output_dir, f'seed{seed}')
        os.makedirs(output_seed_dir, exist_ok=True)
        output_train_json = os.path.join(output_seed_dir, f'train_{idx}.json')
        output_val_json = os.path.join(output_seed_dir, f'val_{idx}.json')

        
        with open(output_train_json, 'w') as train_writer:
            json.dump(train_data, train_writer)
        print(f'done. {output_train_json}')
        
        with open(output_val_json, 'w') as val_writer:
            json.dump(val_data, val_writer)
        print(f'done. {output_val_json}')

if __name__ == "__main__":
    split_type = int(args.split_type)
    input_path = args.input_path
    output_dir_path = args.output_path
    val_ratio = float(args.val_ratio)
    seed = args.seed
    
    seed_everything(seed)
    
    if split_type == 1:
        stratified_group_kfold_dataset(input_path, output_dir_path, seed)
    elif split_type == 2:
        random_split_dataset(input_path, output_dir_path, val_ratio, seed)
    else:
        raise ValueError