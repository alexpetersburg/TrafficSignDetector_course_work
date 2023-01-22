"""
A script to convert self dataset to coco format

Usage:
  Example:  python dataset_to_coco.py -p synth_signs -t synth_signs
  For help: python dataset_to_coco.py -h

"""
import os
import argparse
import yaml
import cv2
import json
import pathlib

COCO_TEMPLATE = {"info": {"description": "", "url": "", "version": "", "year": 2020, "contributor": "", "date_created": ""},
                 "licenses": [{"id": "", "name": "", "url": ""}],
                 "categories": [{"id": "", "name": "", "supercategory": ""}],
                 "images": [{"id": "", "file_name": "", "width": "", "height": "", "date_captured": "", "license": "", "coco_url": "", "flickr_url": ""}],
                 "annotations": [{"id": "", "image_id": "", "category_id": "", "iscrowd": 0, "area": "", "bbox": "", "segmentation": ""}]}


class Params:
    """
    Load yaml project file

    Parametres
    -----
    project_file   - path/to/project/file

    Returns
    -----
    project dict
    """
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    parser.add_argument('-d', '--datasets_path', type=str, default='datasets/', help='Path to folder with datasets')
    parser.add_argument('-t', '--dataset_type', type=str, required=True, help='Type of dataset (synth_signs or gtsdb)')
    args = parser.parse_args()
    return args

def gtsdb_to_coco(opt):
    """
    Convert gtsdb dataset to coco format.
    See README.md for more details

    Parametres
    -----
    opt  - params from argparse

    """
    params = Params(f'projects/{opt.project}.yml')
    gtsdb_annotation = COCO_TEMPLATE.copy()
    #set base info
    gtsdb_annotation['info']['description'] = "The German Traffic Sign Detection Benchmark"
    gtsdb_annotation['licenses'][0] = {"id": 1, "name": None, "url": None}
    #set categories
    categories = []
    gtsdb_map_path = os.path.join(opt.datasets_path, opt.project, 'annotations', 'gtsdb-map.yaml')
    assert os.path.exists(gtsdb_map_path), "gtsdb-map.yaml must be in dataset/annotations folder"
    gtsdb_map = yaml.safe_load(open(gtsdb_map_path).read())
    id = 1
    for key in sorted(gtsdb_map.keys()):
        category = {"id": id, "name": gtsdb_map[key], "supercategory": ""}
        id+=1
        categories.append(category)
    print([str(v) for k, v in gtsdb_map.items()]) # print obj_list to project file
    gtsdb_annotation['categories'] = categories
    #get gtsdb annotations
    gtsdb_gr_rt = {}
    with open(os.path.join(opt.datasets_path, opt.project, 'annotations', 'gt.txt'),'r') as gt_file:
        for line in gt_file:
            filename, x1, y1, x2, y2, category = line.replace('\n', '').split(';')
            category = int(category) + 1
            x, y, width, height = int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)
            if filename in gtsdb_gr_rt:
                gtsdb_gr_rt[filename].append([x, y, width, height, category])
            else:
                gtsdb_gr_rt[filename] = [[x, y, width, height, category]]
    train_annotation = gtsdb_annotation.copy()
    val_annotation = gtsdb_annotation
    #get train filenames
    train_filenames = []
    train_filenames.extend(os.listdir(os.path.join(opt.datasets_path, opt.project, params.train_set)))
    #get val filenames
    val_filenames = []
    new_index = int(train_filenames[-1].split('.')[0])+1
    val_path = os.path.join(opt.datasets_path, opt.project, params.val_set)
    val_imgs = os.listdir(val_path)
    for img in val_imgs:
        if img in train_filenames:
            new_name = f'{new_index:05}.ppm'
            os.rename(os.path.join(val_path, img), os.path.join(val_path, new_name))
            img = new_name
            new_index+=1
        val_filenames.append(img)
    #set images&annotations
    #train set
    images, annotations = get_coco_style_img_and_annot(train_filenames, gtsdb_gr_rt)
    train_annotation['images'] = images
    train_annotation['annotations'] = annotations
    #val set
    images, annotations = get_coco_style_img_and_annot(val_filenames, gtsdb_gr_rt)
    val_annotation['images'] = images
    val_annotation['annotations'] = annotations
    #save to files
    with open(os.path.join(opt.datasets_path, opt.project, 'annotations', f'instances_{params.train_set}.json'), 'w') as outfile:
        json.dump(train_annotation, outfile)
    with open(os.path.join(opt.datasets_path, opt.project, 'annotations', f'instances_{params.val_set}.json'), 'w') as outfile:
        json.dump(val_annotation, outfile)


def synth_to_coco(opt):
    """
    Convert gtsdb dataset to coco format.
    See README.md for more details

    Parametres
    -----
    opt  - params from argparse

    """
    params = Params(f'projects/{opt.project}.yml')
    synth_annotation = COCO_TEMPLATE.copy()
    #set base info
    synth_annotation['info']['description'] = "Synthetic traffic sign dataset"
    synth_annotation['licenses'][0] = {"id": 1, "name": None, "url": None}
    #set categories
    categories = []
    synth_map = yaml.safe_load(open(os.path.join(opt.datasets_path, opt.project, 'annotations', 'labels-DE-169.yaml')).read())
    cat_to_id = {str(v): int(k) for k, v in synth_map.items()}
    print([str(v) for k, v in synth_map.items()]) # print obj_list to project file
    id = 1
    for key in sorted(synth_map.keys()):
        category = {"id": id, "name": str(synth_map[key]), "supercategory": ""}
        id+=1
        categories.append(category)
    synth_annotation['categories'] = categories
    #get annotations
    train_annotation = synth_annotation.copy()
    val_annotation = synth_annotation
    train_gr_rt = get_synth_like_gr_tr_dict(os.path.join(opt.datasets_path, opt.project, params.train_set, 'multiclass.csv'), cat_to_id)
    val_gr_tr = get_synth_like_gr_tr_dict(os.path.join(opt.datasets_path, opt.project, params.val_set, 'multiclass.csv'), cat_to_id)
    #get train filenames
    train_filenames = os.listdir(os.path.join(opt.datasets_path, opt.project, params.train_set,'imgs'))
    #get val filenames
    val_filenames = os.listdir(os.path.join(opt.datasets_path, opt.project, params.val_set,'imgs'))
    #set images&annotations
    #train set
    images, annotations = get_coco_style_img_and_annot(train_filenames, train_gr_rt)
    train_annotation['images'] = images
    train_annotation['annotations'] = annotations
    #val set
    images, annotations = get_coco_style_img_and_annot(val_filenames, val_gr_tr)
    val_annotation['images'] = images
    val_annotation['annotations'] = annotations
    #save to files
    with open(os.path.join(opt.datasets_path, opt.project, 'annotations', f'instances_{params.train_set}.json'), 'w') as outfile:
        json.dump(train_annotation, outfile)
    with open(os.path.join(opt.datasets_path, opt.project, 'annotations', f'instances_{params.val_set}.json'), 'w') as outfile:
        json.dump(val_annotation, outfile)

def get_synth_like_gr_tr_dict(path_to_gr_tr, cat_to_id):
    """
    Load gorund truth(synth style) file.
            paht/to/file/,  x1,y1,x2,y2,  category_name
            ...
    Parametres
    -----
    path_to_gr_tr
    cat_to_id      - dict to map category_name to id

    Returns
    -----
    [[x, y, width, height, int(category)]]
    """
    gr_tr = {}
    with open(path_to_gr_tr,'r') as gt_file:
        for line in gt_file:
            filename, x1, y1, x2, y2, category = line.replace('\n', '').split(',')
            filename = pathlib.Path(filename).name
            category = cat_to_id[category.replace('_filp', '')] + 1
            x, y, width, height = int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)
            if filename in gr_tr:
                gr_tr[filename].append([x, y, width, height, category])
            else:
                gr_tr[filename] = [[x, y, width, height, int(category)]]
    return gr_tr


def get_coco_style_img_and_annot(filenames, gr_tr):
    """
    Create coco style bbox and img lists
    Parametres
    -----
    filenames
    gr_tr        - dict to map filename to bbox annotation

    Return
    -----
    [{"id": , "file_name": }],
    [{"id": , "image_id": , "category_id": , 'area': , 'iscrowd': 0, "bbox": }]
    """
    images = []
    annotations = []
    img_id = 1
    annot_id = 1
    for file in filenames:
        images.append({"id": img_id, "file_name": file})
        if file in gr_tr:
            for bbox in gr_tr[file]:
                annotations.append({"id": annot_id, "image_id": img_id, "category_id": bbox[4],
                                    'area': bbox[3]*bbox[4], 'iscrowd': 0, "bbox": bbox[:4]})
                annot_id+=1
        img_id+=1
    return images, annotations


if __name__ == '__main__':
    opt = get_args()
    if opt.dataset_type=="gtsdb":
        gtsdb_to_coco(opt)
    elif opt.dataset_type=="synth_signs":
        synth_to_coco(opt)
    else:
        print('Unknown dataset type')
