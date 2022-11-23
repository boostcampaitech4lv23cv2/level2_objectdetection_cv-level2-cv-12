from collections import Counter
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedGroupKFold

def get_distribution(y):
    y_distr = Counter(y)
    y_vals_sum = sum(y_distr.values())

    return [f'{y_distr[i]/y_vals_sum:.2%}' for i in range(np.max(y) +1)]


def display_kfold_distribution(input_json):
    with open(input_json) as f: 
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    annotation_infos = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]
    X = np.ones((len(data['annotations']), 1))
    y = np.array([annotation_info[1] for annotation_info in annotation_infos])
    groups = np.array([annotation_info[0] for annotation_info in annotation_infos])

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=41)

    random_seed = 41
    
    distrs = [get_distribution(y)]
    index = ['training set']

    for fold_ind, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        train_y, val_y = y[train_idx], y[val_idx]
        train_gr, val_gr = groups[train_idx], groups[val_idx]

        assert len(set(train_gr) & set(val_gr)) == 0
        
        distrs.append(get_distribution(train_y))
        distrs.append(get_distribution(val_y))
        index.append(f'train - fold{fold_ind}')
        index.append(f'val - fold{fold_ind}')

    categories = [d['name'] for d in data['categories']]
    pd.DataFrame(distrs, index=index, columns = [categories[i] for i in range(np.max(y) + 1)])