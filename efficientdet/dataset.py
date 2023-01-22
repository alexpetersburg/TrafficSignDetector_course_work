import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
import albumentations as A
import random

class CocoDataset(Dataset):
    def __init__(self, root_dir, set_name='train2017', transform=None):

        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform
        dataset_path = os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json')
        assert os.path.exists(dataset_path), "/path/to/annotations does not exist"
        self.coco = COCO(dataset_path)
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image
        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Augmentation."""
    def __init__(self):
        self.image_aug = A.Compose(
        [
            A.OneOf(
                [
                 A.RandomRain(p=0, brightness_coefficient=0.9, drop_width=1, blur_value=3),
                 A.RandomSnow(p=0, brightness_coeff=1.5, snow_point_lower=0.5, snow_point_upper=1),
                 A.RandomFog(p=0.2, fog_coef_lower=0.09, fog_coef_upper=0.1, alpha_coef=0.3),
                 A.Compose([
                    A.MotionBlur(p=0.4, blur_limit=4),
                    A.GaussNoise(p=0.4)], p = 1)
        ], p = 0.9),
            A.OneOf(
                [
                 A.RandomContrast(),
                 A.RandomGamma(),
                 A.RandomBrightness(),], p = 0.1),
        ], p = 0.3)
        aug_resizer = A.Compose(
        [
            A.OneOf([
                 RandomSizedBBoxSafeCropModified(1360, p=0),
                 ExtendRandomSize(p=1, min_bbox_size=45)
        ],p=1),
            A.Rotate(5,p=0,border_mode=0),
            RandomResize(max_scale=0, min_bbox_size=45, p=0.3)

        ],p=0.8)
        self.bbox_aug = A.Compose([aug_resizer],
                                bbox_params=A.BboxParams(format='pascal_voc', min_area=0.,
                                min_visibility=0.))


    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        annotations = {'image': image, 'bboxes': list(annots)}
        augmented = self.bbox_aug(**annotations)
        annots = np.reshape(augmented['bboxes'],(-1,5))
        image = self.image_aug(image=augmented['image'])['image']
        return {'img': image, 'annot': annots}


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}


class ExtendRandomSize(A.DualTransform):
    """Scale image by zero padding."""

    def __init__(self, min_bbox_size=26, p=1, always_apply=False):
        super(ExtendRandomSize, self).__init__(always_apply, p)
        self.min_bbox_size = min_bbox_size

    def apply(self, img, height=0, width=0, old_height=0, old_width=0, **params):
        new_image = np.zeros((int(old_height), int(old_width), 3),dtype= np.float32)
        scaled_img = A.augmentations.functional.resize(img, height, width)
        new_image[0:height, 0:width] = scaled_img
        return new_image

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]
        if len(params["bboxes"]) == 0:  # less likely, this class is for use with bboxes.
            return {"height": img_h, "width": img_w, "old_height": img_h, "old_width": img_w}
        #compute scale
        bboxes = A.augmentations.bbox_utils.denormalize_bboxes(params["bboxes"], img_h, img_w)
        min_bbox = min([bbox[2]-bbox[0] for bbox in bboxes])
        if min_bbox<self.min_bbox_size:
            return {"height": img_h, "width": img_w, "old_height": img_h, "old_width": img_w}
        max_scale = min_bbox/self.min_bbox_size
        scale = random.uniform(1, max_scale)
        #zero padding
        return {"height": int(img_h/scale), "width": int(img_w/scale), "old_height": img_h, "old_width": img_w}
    def apply_to_bbox(self, bbox, height=0, width=0, old_height=0, old_width=0, **params):
        bbox = A.augmentations.bbox_utils.denormalize_bbox(bbox, old_height, old_width)
        scale = height/old_height
        bbox = [x*scale for x in bbox]
        bbox = A.augmentations.bbox_utils.normalize_bbox(bbox, old_height, old_width)
        return bbox

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]


class RandomResize(A.DualTransform):
    """Random stretch image"""

    def __init__(self, max_scale=0.2, min_bbox_size=26, p=1, always_apply=False):
        super(RandomResize, self).__init__(always_apply, p)
        self.max_scale = max_scale
        self.min_bbox_size = min_bbox_size

    def apply(self, img, height=0, width=0, **params):
        return A.augmentations.functional.resize(img, height=height, width=width)

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]
        if len(params["bboxes"]) == 0:  # less likely, this class is for use with bboxes.
            return {"height": img_h, "width": img_w}
        #compute scale
        bboxes = A.augmentations.bbox_utils.denormalize_bboxes(params["bboxes"], img_h, img_w)
        min_bbox = min([bbox[2]-bbox[0] for bbox in bboxes])
        if min_bbox<self.min_bbox_size:
            return {"height": img_h, "width": img_w}
        max_scale = min(self.max_scale, (min_bbox/self.min_bbox_size)-1)
        scale_high = random.uniform(1-max_scale, 1+max_scale)
        scale_widh = random.uniform(1-max_scale, 1+max_scale)
        #zero padding
        return {"height": int(img_h*scale_high), "width": int(img_w*scale_widh)}
    def apply_to_bbox(self, bbox, **params):
        return bbox

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]


class RandomSizedBBoxSafeCropModified(A.DualTransform):
    """Crop a random part of the input and rescale it to some size without loss of bboxes.

    Args:
        height (int): height after crop and resize.
        width (int): width after crop and resize.
        erosion_rate (float): erosion rate applied on input image height before crop.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes

    Image types:
        uint8, float32
    """

    def __init__(self, max_size=1300, erosion_rate=0.0, interpolation=cv2.INTER_LINEAR, always_apply=False, p=1.0):
        super(RandomSizedBBoxSafeCropModified, self).__init__(always_apply, p)
        self.interpolation = interpolation
        self.erosion_rate = erosion_rate
        self.max_size = A.LongestMaxSize(max_size=max_size)

    def apply(self, img, crop_height=0, crop_width=0, h_start=0, w_start=0, interpolation=cv2.INTER_LINEAR, **params):
        crop = A.augmentations.functional.random_crop(img, crop_height, crop_width, h_start, w_start)
        return self.max_size.apply(crop)

    def get_params_dependent_on_targets(self, params):
        img_h, img_w = params["image"].shape[:2]
        if len(params["bboxes"]) == 0:  # less likely, this class is for use with bboxes.
            erosive_h = int(img_h * (1.0 - self.erosion_rate))
            crop_height = img_h if erosive_h >= img_h else random.randint(erosive_h, img_h)
            return {
                "h_start": random.random(),
                "w_start": random.random(),
                "crop_height": crop_height,
                "crop_width": int(crop_height * img_w / img_h),
            }
        # get union of all bboxes
        x, y, x2, y2 = A.augmentations.bbox_utils.union_of_bboxes(
            width=img_w, height=img_h, bboxes=params["bboxes"], erosion_rate=self.erosion_rate
        )
        # find bigger region
        bx, by = x * random.random(), y * random.random()
        bx2, by2 = x2 + (1 - x2) * random.random(), y2 + (1 - y2) * random.random()
        bw, bh = bx2 - bx, by2 - by
        crop_height = img_h if bh >= 1.0 else int(img_h * bh)
        crop_width = img_w if bw >= 1.0 else int(img_w * bw)
        h_start = np.clip(0.0 if bh >= 1.0 else by / (1.0 - bh), 0.0, 1.0)
        w_start = np.clip(0.0 if bw >= 1.0 else bx / (1.0 - bw), 0.0, 1.0)
        return {"h_start": h_start, "w_start": w_start, "crop_height": crop_height, "crop_width": crop_width}

    def apply_to_bbox(self, bbox, crop_height=0, crop_width=0, h_start=0, w_start=0, rows=0, cols=0, **params):
        bbox = A.augmentations.functional.bbox_random_crop(bbox, crop_height, crop_width, h_start, w_start, rows, cols)
        bbox = self.max_size.apply_to_bbox(bbox, **params)
        return bbox

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def get_transform_init_args_names(self):
        return ("height", "width", "erosion_rate", "interpolation")
