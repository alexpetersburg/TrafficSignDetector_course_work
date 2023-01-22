import argparse
import os, sys
import cv2
import copy
import json
import numpy as np
np.random.bit_generator = np.random._bit_generator
from tqdm import tqdm
import imgaug.augmenters as iaa
import imgaug
import utils
import random
print(imgaug.__version__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Filter objects and convert it into coco json annotation format')
    parser.add_argument(
        '--path_to_mapillary_data',
        help='path to mapillary dataset',
        default='/media/hada_admin/2A4646D266EE18A2/datasets/mapillary-vistas-dataset_public_v1.1',
        type=str
    )
    parser.add_argument(
        '--sign_templates',
        help='path to folder with templates',
        default='/home/hada_admin/syntetic_signs/templates',
        type=str
    )
    parser.add_argument(
        '--targ_img_w',
        help='target image width',
        default=1360,
        type=str
    )
    parser.add_argument(
        '--targ_img_h',
        help='target image height',
        default=800,
        type=str
    )
    parser.add_argument(
        '--out_dir',
        dest='out_dir',
        help='path to detectron output images',
        default='/media/hada_admin/2A4646D266EE18A2/datasets/mapillary_with_syntetic_signs',
        type=str
    )
    parser.add_argument(
        '--only_boxes',
        dest='only_boxes',
        help='dump segmentation or not',
        default=True,
        type=str
    )
    parser.add_argument(
        '--saving_interval',
        help='dump interval',
        default=100,
        type=int
    )
    return parser.parse_args()


def read_mapillary_instances_from_config(path, target_classnames=('Traffic Sign (Front)')):
    """
    Read mapillary config to get info about colors of target clusters
    Args:
        path: path to config
        target_classnames: readable names of target classes
    Returns: Info about target classes

    """
    out_classes = []
    with open(os.path.join(path, 'config.json')) as config_file:
        config = json.load(config_file)
    for label in config["labels"]:

        if label["instances"] and label["readable"] in target_classnames:
            out_classes.append(label)
    return out_classes


def mapillary_id_by_classname(classname, cats):
    """
    Return category ID by classname
    Args:
        classname: str with classname
        cats: dict with categories info
    Returns: id of category (int)

    """
    for category_info in cats:
        if classname in category_info['mapillary_cats']:
            return category_info['id']


def get_classname_by_color(color, instances):
    """

    Args:
        color: object color
        instances: info about target classes
    Returns: name of corresponding class

    """
    classname = None
    for instance in instances:
        if instance["color"] == color:
            return instance["readable"]
    return classname


def get_contours(mask):
    """
    Get cv2 contours from binary mask
    """
    ret, thresh = cv2.threshold(mask, 2, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_bbox(contours):
    """
    Get bbox around all object contours
    """
    tmp_rect = cv2.boundingRect(contours[0])
    left = tmp_rect[0]
    top = tmp_rect[1]
    right = tmp_rect[0] + tmp_rect[2]
    bottom = tmp_rect[1] + tmp_rect[3]
    for contour in contours:
        tmp_rect = cv2.boundingRect(contour)
        if tmp_rect[0] < left:
            left = tmp_rect[0]
        if tmp_rect[1] < top:
            top = tmp_rect[1]
        if tmp_rect[0] + tmp_rect[2] > right:
            right = tmp_rect[0] + tmp_rect[2]
        if tmp_rect[1] + tmp_rect[3] > bottom:
            bottom = tmp_rect[1] + tmp_rect[3]
    return [float(left), float(top), float(right - left), float(bottom - top)]


def filter_polys(polys):
    """
    Filter contours with invalid prroperties
    """
    out_polys = []
    for poly in polys:
        if len(poly) > 6:
            out_polys.append(poly)
    return out_polys


def get_random_sign(templates_info, proportion, prop_diff=0.2):
    """
    Get random sign from templates_info which close by proportion (dx/dy) to target region on image
    Args:
        templates_info: information about signs templates
        proportion: target position proportion (dx / dy)
        prop_diff: max difference between proportions
    Returns:
        Random sing template and it's ID
    """
    order = np.random.choice(len(templates_info), size=len(templates_info), replace=False)
    for idx in order:
        if abs(templates_info[idx]['proportion'] - proportion) < prop_diff:
            return templates_info[idx], templates_info[idx]['category']
    random_id = np.random.randint(0, len(templates_info))
    return templates_info[random_id], templates_info[random_id]['category']


def insert_sign_template_to_position(img, target_rect, sign_template, min_target_size):
    """
    Insert sign template into target position (use imgaug to make little augment sign template)
    Args:
        img: target rgb image
        target_rect: position to insert
        sign_template: sign template
    Returns:
        Modified image
    """
    #estimate template scale using rect and template height
    max_scale = target_rect[3] / sign_template['img'].shape[0]
    min_scale = min_target_size / sign_template['img'].shape[0]
    scale = random.uniform(min_scale, max_scale)
    center_pos = [int(target_rect[0] + 0.5 * target_rect[2]), int(target_rect[1] + 0.5 * target_rect[3])]
    pos_to_insert = [max(int(center_pos[0] - scale * sign_template['img'].shape[1]/2), 0),
                     max(int(center_pos[1] - scale * sign_template['img'].shape[0]/2), 0),
                     min(int(center_pos[0] + scale * sign_template['img'].shape[1]/2), img.shape[1] - 1),
                     min(int(center_pos[1] + scale * sign_template['img'].shape[0]/2), img.shape[0] - 1)]
    scaled_target = cv2.resize(cv2.cvtColor(sign_template['img'], cv2.COLOR_RGBA2RGB),
                               (pos_to_insert[2] - pos_to_insert[0], pos_to_insert[3] - pos_to_insert[1]))
    scaled_mask = cv2.resize(sign_template['mask'],
                             (pos_to_insert[2] - pos_to_insert[0], pos_to_insert[3] - pos_to_insert[1]))
    shape = iaa.Sequential([iaa.PerspectiveTransform((0.01, 0.15)),
                          iaa.Affine(rotate=(-10, 10), shear=(-5, 5))])
    shape = shape.localize_random_state()
    seq_mask = shape.to_deterministic()
    seq_img = shape.to_deterministic()
    seq_mask = seq_mask.copy_random_state(seq_img, matching="name")
    scaled_mask = seq_mask(image=scaled_mask)
    seq_img.append(iaa.GammaContrast((0.5, 2.0)))
    seq_img.append(iaa.MultiplyBrightness((0.25, 1.)))
    seq_img.append(iaa.OneOf([
                    iaa.AdditiveGaussianNoise(0.05*255),
                    iaa.GaussianBlur(1),
                    iaa.AverageBlur(k=2),
                    iaa.MotionBlur(k=(3,6))
                ]))
    if min(scaled_mask.shape[:2])>=32:
        seq_img.append(iaa.Sometimes(
                   0.3,
                   iaa.imgcorruptlike.Spatter(severity=(1,5))))

    scaled_target = seq_img(image=scaled_target)
    roi = img[pos_to_insert[1]:pos_to_insert[3], pos_to_insert[0]:pos_to_insert[2]]
    roi[scaled_mask > 128] = scaled_target[scaled_mask > 128]
    img[pos_to_insert[1]:pos_to_insert[3], pos_to_insert[0]:pos_to_insert[2]] = roi
    return img, [pos_to_insert[0],
                 pos_to_insert[1],
                 pos_to_insert[2] - pos_to_insert[0],
                 pos_to_insert[3] -pos_to_insert[1]]


def process_mapillary_sample(path_to_mapillary_data, dataset, img_name, instances, templates_info, image_id=1,
                        last_obj_id=0, target_w=1920, target_h=1080, only_boxes=False, min_target_size=30):
    """
    Replase original signs to templates on single mapillary image
    Args:
        path_to_mapillary_data: path to mapillary dataset root
        dataset: "training" or "validation"
        img_name: name of target image
        instances: information about target classes
        templates_info: information about signs templates
        image_id: image id (needs for COOO-like annotation)
        last_obj_id: id of last object (needs for COOO-like annotation)
        target_w: scale image width to target resolution (no scalling if target_w=None)
        target_h: scale image height to target resolution (no scalling if target_h=None)
        only_boxes: dump only boxes annotation (dont put to annotation information about contours)
        min_target_size: minimal size of replaaced source sign bbox(min(dx, dy))

    Returns: Image annotation, modified source image and id of last objet

    """
    inst_img = cv2.imread(os.path.join(path_to_mapillary_data, dataset, "instances", img_name[:-3] + "png"))
    pan_img = cv2.imread(os.path.join(path_to_mapillary_data, dataset, "panoptic", img_name[:-3] + "png"))
    label_img = cv2.imread(os.path.join(path_to_mapillary_data, dataset, "labels", img_name[:-3] + "png"))
    img = cv2.imread(os.path.join(path_to_mapillary_data, dataset, "images", img_name))

    image_annotations = []

    if target_h is not None and target_w is not None:
        inst_img = cv2.resize(inst_img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        pan_img = cv2.resize(pan_img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        label_img = cv2.resize(label_img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    for idx in np.unique(inst_img):
        temp_mask = copy.deepcopy(inst_img)
        temp_mask[inst_img != idx] = 0
        temp_mask[inst_img == idx] = 255
        color = [int(label_img[temp_mask == 255][2]),
                 int(label_img[temp_mask == 255][1]),
                 int(label_img[temp_mask == 255][0])]
        classname = get_classname_by_color(color, instances)
        if classname is not None:
            tmp_pan_img = copy.deepcopy(pan_img)
            panoptic_array = np.array(tmp_pan_img).astype(np.uint32)
            panoptic_id_array = panoptic_array[:, :, 0] + (2 ** 8) * panoptic_array[:, :, 1] + (
                        2 ** 16) * panoptic_array[:, :, 2]
            panoptic_id_array[cv2.cvtColor(temp_mask, cv2.COLOR_BGR2GRAY) == 0] = 1234567
            panoptic_ids_from_image = np.unique(panoptic_id_array)

            for pan_id in panoptic_ids_from_image:
                if pan_id != 1234567:
                    bad_contour = False
                    tmp_pan_img = copy.deepcopy(panoptic_id_array)
                    tmp_pan_img[tmp_pan_img != pan_id] = 0
                    tmp_pan_img[tmp_pan_img == pan_id] = 255
                    contours = get_contours(tmp_pan_img.astype(np.uint8))
                    #inpaint original sign
                    img = cv2.inpaint(img, tmp_pan_img.astype(np.uint8), 3, cv2.INPAINT_NS)
                    segmentation = [c.reshape(-1).tolist() for c in contours]
                    segmentation = filter_polys(segmentation)
                    if segmentation == []:
                        bad_contour = True
                    rect = get_bbox(contours)
                    if min(rect[2], rect[3]) < min_target_size:
                        continue
                    sign_to_insert, targ_class_id = get_random_sign(templates_info, rect[2] / rect[3])
                    img, rect = insert_sign_template_to_position(img, rect, sign_to_insert, min_target_size=min_target_size)
                    if not bad_contour:
                        last_obj_id += 1
                        seg_to_dump = []
                        if not only_boxes:
                            seg_to_dump = list(segmentation)
                        annotation = {
                            'segmentation': seg_to_dump,
                            'iscrowd': 0,
                            'image_id': image_id,
                            'category_id': int(targ_class_id),
                            'id': last_obj_id,
                            'bbox': list(rect),
                            'area': rect[2] * rect[3]
                        }
                        image_annotations.append(annotation)
    return image_annotations, last_obj_id, img


def vis_annotation(image_annotations, img):
    """
    Visualize image annotation

    """
    out_img = copy.deepcopy(img)
    for annotation in image_annotations:
        for k in range(len(annotation["segmentation"])):
            for i in range(0, len(annotation["segmentation"][k]) - 1, 2):
                cv2.circle(out_img, (int(annotation["segmentation"][k][i]), int(annotation["segmentation"][k][i + 1])),
                           3, (255, 255, 255), -11)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(out_img,
                      (int(annotation["bbox"][0]), int(annotation["bbox"][1])),
                      (int(annotation["bbox"][0]) + int(annotation["bbox"][2]),
                       int(annotation["bbox"][1] + annotation["bbox"][3])),
                      (255, 0, 0), 3)
        cv2.putText(out_img, 'ID: ' + str(annotation["category_id"]) + ", #" + str(annotation["id"]),
                    (int(annotation["bbox"][0]) + 10, int(annotation["bbox"][1]) + 20),
                    font, 1., (0, 0, 255), 1, cv2.LINE_AA)
    return out_img


def read_sign_templates(path_to_templates):
    """
    Read sign templates
    """
    templates = []
    categories = []
    sample_names = os.listdir(path_to_templates)
    id = 1
    for sample_name in sorted(sample_names):
        tmp_id = id
        img = cv2.imread(os.path.join(path_to_templates, sample_name),
                                         cv2.IMREAD_UNCHANGED)
        image, mask = img[:, :, :3], img[:, :, 3]
        image, mask = utils.remove_padding(image, mask)
        name = sample_name[:-4]
        if 'flip' in name:
            name = name.replace('_flip', '')
            tmp_id = next(item['id'] for item in categories if item["name"] == name)
        else:
            categories.append({'id': id, 'name': name})
            id += 1
        templates.append({'name': name,
                         'img': image,
                         'mask': mask,
                         'proportion': mask.shape[1] / mask.shape[0],
                         'category': tmp_id})

    return templates, categories


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    out_dir_annotations = os.path.join(args.out_dir, 'annotations')
    os.makedirs(out_dir_annotations, exist_ok=True)
    out_dir_temp_annotations = os.path.join(args.out_dir, 'temp_annos')
    os.makedirs(out_dir_temp_annotations, exist_ok=True)

    instances = read_mapillary_instances_from_config(args.path_to_mapillary_data)
    templates_info, categories = read_sign_templates(args.sign_templates)
    print('Categories: ', list([t['name'] for t in categories]))
    datasets = ['training', 'validation']
    for dataset in datasets:
        image_idx = 1
        last_obj_id = 0
        ann_dict = {}
        images = []
        annotations = []
        img_list = os.listdir(os.path.join(args.path_to_mapillary_data, dataset, "images"))
        img_list.sort()
        out_dir_images = os.path.join(args.out_dir, dataset)
        os.makedirs(out_dir_images, exist_ok=True)
        image_counter = 0
        for img_name in tqdm(img_list, desc='Converting Mapillary ' + dataset):
            image_annos, last_obj_id, targ_img = process_mapillary_sample(args.path_to_mapillary_data,
                                                                          dataset,
                                                                          img_name,
                                                                          instances,
                                                                          templates_info,
                                                                          image_id=image_idx,
                                                                          last_obj_id=last_obj_id,
                                                                          target_w=args.targ_img_w,
                                                                          target_h=args.targ_img_h,
                                                                          only_boxes=args.only_boxes)
            if len(image_annos) > 0:
                for anno in image_annos:
                    annotations.append(anno)
                image = {}

                h, w, c = targ_img.shape
                image['id'] = image_idx
                image['width'] = w
                image['height'] = h
                image['file_name'] = img_name
                image['seg_file_name'] = img_name[:-3] + "png"
                images.append(image)
                image_idx += 1
                cv2.imwrite(os.path.join(out_dir_images, img_name), targ_img)
                out_img = vis_annotation(image_annos, targ_img)
                cv2.imshow('frame',  cv2.resize(out_img, (1280, 720)))
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            if image_idx % args.saving_interval == 0:
                tmp_ann_dict = {}
                tmp_ann_dict['images'] = images
                tmp_ann_dict['categories'] = categories
                tmp_ann_dict['annotations'] = annotations
                if not os.path.exists(os.path.join(out_dir_temp_annotations, dataset)):
                    os.mkdir(os.path.join(out_dir_temp_annotations, dataset))
                with open(os.path.join(out_dir_temp_annotations, dataset,  dataset + str(
                          image_idx) + ".json"), 'w') as outfile:
                    outfile.write(json.dumps(tmp_ann_dict, sort_keys=True, indent=4))
            image_counter += 1

        ann_dict['images'] = images
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations
        with open(os.path.join(out_dir_annotations, "instances_" + dataset + ".json"), 'w') as outfile:
             outfile.write(json.dumps(ann_dict, sort_keys=True, indent=4))
