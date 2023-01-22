"""
Using this inference script, you can detect traffic signs
for each frame from the video.


Output
-----
csv for each video with bbox predictions for each frame

Usage:
  Example:  python inference.py -p synth_signs -c 3 -i input -o output -w logs\synth_signs\efficientdet-d3_0_1.pth
  For help: python inference.py -h

"""

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import invert_affine, postprocess
from backbone import EfficientDetBackbone
from efficientdet.dataset import Resizer, Normalizer
import os
from pathlib import Path
import cv2
import json
import yaml
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import IterableDataset
import argparse
import traceback
import math
from utils.sort import *
import time
import operator

INPUT_SIZES = [512, 640, 768, 896, 1024, 1280, 1280, 1536]


class IteratorFromClip(object):
    """
    This class creates an iterator from a part of the video.
    On iteration, returns dict:
                {'image': frame, 'frame_timestamp': frame_timestamp, 'scale':scale}
    """
    def __init__(self, video_path, start_frame, num_of_iterations, transform):
        """
        Init iterator

        Parametres
        -----
        video_path            - The path to the video
        start_frame           - The index of the frame at which part of
                                the video begins
        num_of_iterations     - The number of iterations for the video clip
        transform             - torchvision.transforms for frames

        """
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.num_of_iterations = num_of_iterations
        self.transform = transform
    def __iter__(self):
        """
        Returns iterator
        """
        return self



    def __next__(self):
        """
        While num_of_iterations>0 returns:
                    {'image': frame, 'frame_timestamp': frame_timestamp, 'scale':scale}

        """
        if self.num_of_iterations>0:
            frame_num = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            frame_timestamp = humanize_time(1000*frame_num / self.fps )
            ret, frame = self.cap.read()
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.
            scale = 1.
            if self.transform:
                transforms = self.transform(frame)
                frame = transforms['image']
                scale = transforms.get('scale')
                scale = scale if scale is not None else 1.
            self.num_of_iterations-=1
            return {'image': frame, 'frame_timestamp': frame_timestamp, 'scale':scale}
        else:
            self.cap.release()
            raise StopIteration()


class VideoDataset(IterableDataset):
    """
    This class is an IterableDataset of video.
    On iteration, returns dict:
                    {'image': frame, 'frame_timestamp': frame_timestamp, 'scale':scale}
    """
    def __init__(self, video_path, transform=None):
        """
        Init dataset

        Parametres
        -----
        video_path            - The path to the video
        transform             - torchvision.transforms for frames

        """
        super(VideoDataset).__init__()
        cap = cv2.VideoCapture(video_path)
        self.video_path = video_path
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.transform = transform


    def __iter__(self):
        """
        Return IteratorFromClip.
        if num_workers>0: Splits video into clips and returns iterators

        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            start_frame = 0
            num_of_iterations = self.frame_count
        else:
            num_of_iterations = int(math.ceil(self.frame_count / worker_info.num_workers))
            worker_id = worker_info.id
            start_frame = worker_id * num_of_iterations
            num_of_iterations = min(self.frame_count - start_frame, num_of_iterations)
        return IteratorFromClip(video_path=self.video_path,
                                  start_frame=start_frame,
                                  num_of_iterations=num_of_iterations,
                                  transform=self.transform )


class Inference_Transformer(object):
    """
    This class resize and normalize image.
    """
    def __init__(self, img_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        Init transformer
        Parametres
        -----
        img_size        - target ш=image size
        mean            - mean deviation
        std             - standard deviation

        """
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])
        self.img_size = img_size

    def __call__(self, image):
        """
        Apply transforms
        Parametres
        -----
        image

        Returns
        -----
        image
        scale  - resize scale
        """
        image = (image.astype(np.float32) - self.mean) / self.std
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

        return {'image': torch.from_numpy(new_image).to(torch.float32), "scale": scale}


def inference_collater(data):
    """
    merges a list of samples to form a mini-batch of Tensors
    """
    imgs = [s['image'] for s in data]
    frame_timestamp = [s['frame_timestamp'] for s in data]
    scales = [s['scale'] for s in data]
    imgs = torch.from_numpy(np.stack(imgs, axis=0))
    imgs = imgs.permute(0, 3, 1, 2)
    return {'image': imgs, 'frame_timestamp': frame_timestamp,'scale':scales}


def humanize_time(ms):
    """
    Convert time in ms to 'hh:mm:ss:ms'

    Parametres
    -----
    ms       -time in ms

    Returns
    -----
    'hh:mm:ss:ms'
    """
    secs, ms = divmod(ms, 1000)
    mins, secs = divmod(secs, 60)
    hours, mins = divmod(mins, 60)
    return '%02d:%02d:%02d:%03d' % (hours, mins, secs, ms)


def detect(model, loader, gpu_params, thresholds):
    """
    Predicts all images in the loader using the model

    Parametres
    -----
    model        - EfficientDetBackbone model
    loader       - torch.utils.data.DataLoader
    gpu_params   - {
                    'use_cuda': ,      - bool
                    'gpu': ,           - int
                    'use_float16':     - bool
                   }
    thresholds   - {
                    'nms_threshold': , - nms threshold
                    'threshold':       - threshold of classification
                   }
    Returns
    -----
    [{'frame_timestamp': frame_timestamp,'detection':[{'category_id': ,'score': , 'bbox':}] }]
    """
    results = []
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    with torch.no_grad():
        for data in loader:
            try:
                x = data['image']
                if gpu_params['use_cuda']:
                    x = x.cuda(gpu_params['gpu'])
                    if gpu_params['use_float16']:
                        x = x.half()
                    else:
                        x = x.float()
                else:
                    x = x.float()
                features, regression, classification, anchors = model(x)
                preds = postprocess(x,
                                    anchors, regression, classification,
                                    regressBoxes, clipBoxes,
                                    thresholds['threshold'], thresholds['nms_threshold'])

                preds = invert_affine(data['scale'], preds)
                frame_timestamp = data['frame_timestamp']
                for i in range(len(preds)):
                    if not preds[i]:
                        results.append({'frame_timestamp': frame_timestamp[i],'detection':[]})
                        continue
                    scores = preds[i]['scores']
                    class_ids = preds[i]['class_ids']
                    rois = preds[i]['rois']
                    frame_timestamp[i]
                    if rois.shape[0] > 0:
                        # x1,y1,x2,y2 -> x1,y1,w,h
                        rois[:, 2] -= rois[:, 0]
                        rois[:, 3] -= rois[:, 1]

                        bbox_score = scores
                        image_results = []
                        for roi_id in range(rois.shape[0]):
                            score = float(bbox_score[roi_id])
                            label = int(class_ids[roi_id])
                            box = rois[roi_id, :]
                            image_result = {
                                'category_id': label + 1,
                                'score': float(score),
                                'bbox': box.tolist(),
                                }
                            image_results.append(image_result)
                        results.append({'frame_timestamp': frame_timestamp[i],'detection':image_results})
                    else:
                        results.append({'frame_timestamp': frame_timestamp[i],'detection':[]})

            except Exception as e:
                print('[Error]', traceback.format_exc())
                break
    return results



def make_detection_by_video(video_path, batch_size, num_workers, gpu_params, model,
                        compound_coef, params, thresholds, out_path, tracker):
    """
    Predicts scene and weather of each frame from the video.
    Smooth all predictions with sliding window.

    Parametres
    -----
    video_path              - The path to the video
    batch_size              - How many samples per batch to load.
    num_workers             - How many subprocesses to use for data loading.
    gpu_params              - {
                               'use_cuda': ,      - bool
                               'gpu': ,           - int
                               'use_float16':     - bool
                              }
    model                   - EfficientDetBackbone model
    compound_coef           - coefficients of efficientdet
    params                  - project file (Must include mean and std)
    thresholds              - {
                               'nms_threshold': , - nms threshold
                               'threshold':       - threshold of classification
                              }
                              tracker = {'apply_tracker': args.tracking, 'max_age': args.max_age, 'min_hits': args.min_hits}
    tracker                 - {
                               'apply_tracker': ,      - bool
                               'max_age': ,            - int
                               'min_hits':             - int
                              }
    out_path                - Path to output folder with csv for each video
                              with predictions for each frame


    Output
    -----
    csv with predictions for each frame
    """
    video_name = os.path.split(video_path)[1]
    data = VideoDataset(video_path, transform=Inference_Transformer(img_size=INPUT_SIZES[compound_coef],
                                                                    mean=params['mean'],
                                                                    std=params['std']))
    loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                            num_workers=num_workers, collate_fn=inference_collater)

    results = detect(model, loader, gpu_params= gpu_params, thresholds= thresholds)
    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')
    results = sorted(results, key = lambda i: i['frame_timestamp'])
    if tracker['apply_tracker'] and len(results)>0:
        tracker = Sort(min_hits= tracker['min_hits'], max_age= tracker['max_age'])
        tracker_results = []
        class_freq_per_id = {}
        print('Applying tracking...', end='\t')
        for i in range(len(results)): # apply tracking
            timestamp, detections = results[i]['frame_timestamp'], results[i]['detection']
            if len(detections)>0:
                dets = [detection['bbox'] + [detection['score']] + [detection['category_id']] for detection in detections]
                dets = np.array(dets)
                dets[:, 2:4] += dets[:, 0:2]
                tracker_result = tracker.update(dets)
                tracker_result[:, 2:4] -= tracker_result[:, 0:2]
            else:
                tracker_result = tracker.update()
            if len(tracker_result)>0:
                detection = []
                for track in tracker_result:
                    tr_to_detection = {'bbox':track[:4].tolist(),
                                       'score':float(track[4]),
                                       'category_id':int(track[-1]),
                                       'obj_id': int(track[-2])}
                    if class_freq_per_id.get(tr_to_detection['obj_id']):
                        if class_freq_per_id[tr_to_detection['obj_id']].get(tr_to_detection['category_id']):
                            class_freq_per_id[tr_to_detection['obj_id']][tr_to_detection['category_id']]+= 1
                        else:
                            class_freq_per_id[tr_to_detection['obj_id']][tr_to_detection['category_id']] = 1
                    else:
                        class_freq_per_id[tr_to_detection['obj_id']] = {tr_to_detection['category_id']: 1}
                    detection.append(tr_to_detection)
                tracker_results.append({'frame_timestamp':timestamp,
                                        'detection': detection})
            else:
                tracker_results.append({'frame_timestamp':timestamp,
                                        'detection': []})
        id_to_class = {obj_id: max(class_freq_dict.items(), key=operator.itemgetter(1))[0] for obj_id, class_freq_dict in class_freq_per_id.items()}
        print()
        results = [{'frame_timestamp': tr['frame_timestamp'],
                    'detection':[{'bbox':det['bbox'],
                                  'score': det['score'],
                                  'category_id': id_to_class[det['obj_id']],
                                  'obj_id': det['obj_id']} for det in tr['detection']]}
                    for tr in tracker_results]
    out_filename = f'{video_name}.json'
    json.dump(results, open(os.path.join(out_path, out_filename), 'w'), indent=4)
    print('Complete')


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--in_path', help='Path to the video or \
                                                 path to the folder with videos')
    parser.add_argument('-o', '--out_path', help="Path to output folder")
    parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-w', '--weights', type=str, default=None, help='/path/to/weights')
    parser.add_argument('--nms_threshold', type=float, default=0.5, help='nms threshold, don\'t change it if not for testing purposes')
    parser.add_argument('--threshold', type=float, default=0.05, help='threshold of classification, don\'t change it if not for testing purposes')
    parser.add_argument('--cuda', type=boolean_string, default=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--float16', type=boolean_string, default=False)
    parser.add_argument('--batch_size', type=int, default=4, help='The number of images per batch among all devices')
    parser.add_argument('-n', '--num_workers', type=int, default=12, help='num_workers of dataloader')
    parser.add_argument('--tracking', type=boolean_string, default=False, help='Apply tracker')
    parser.add_argument('--max_age', type=int, default=45, help='Max frames to keep object id without bbox')
    parser.add_argument('--min_hits', type=int, default=15, help='Min frames with object set id')
    parser.add_argument('--project_folder', type=str, default='projects', help='path/to/projects/folder')
    parser.add_argument('--args_from_project', type=boolean_string, default=False, help='get/args/from/project_file')
    args = parser.parse_args()
    params = yaml.safe_load(open(f'{args.project_folder}/{args.project}.yml'))
    if args.args_from_project:
        for key in list(vars(args).keys()):
            if key in params:
                setattr(args, key, params[key])
    return args

if __name__=='__main__':
    args = get_args()

    compound_coef = args.compound_coef
    thresholds = {'nms_threshold': args.nms_threshold, 'threshold': args.threshold}
    gpu_params = {'use_cuda': args.cuda, 'gpu': args.device, 'use_float16': args.float16}
    project_name = args.project
    project_folder = args.project_folder
    params = yaml.safe_load(open(f'{project_folder}/{project_name}.yml'))
    weights_path = args.weights
    in_path = args.in_path
    out_path = args.out_path
    tracker = {'apply_tracker': args.tracking, 'max_age': args.max_age, 'min_hits': args.min_hits}

    print(f'running inference on project {project_name}, weights {weights_path}...')

    obj_list = params['obj_list']

    Path(out_path).mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size
    num_workers= args.num_workers
    if not os.path.exists(weights_path):
        weights_path = os.path.join('weights', weights_path)
    assert os.path.exists(weights_path), "/path/to/weights does not exist"
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                 ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.requires_grad_(False)
    model.eval()
    if gpu_params['use_cuda']:
        model.cuda(gpu_params['gpu'])
        if gpu_params['use_float16']:
            model.half()

    if os.path.isdir(in_path):
        video_names = os.listdir(in_path)
    else:
        in_path, video_name = os.path.split(in_path)
        video_names = [video_name]
    for video_name in video_names:
        print(f'Processing {video_name}...', end='\t')
        make_detection_by_video(video_path= os.path.join(in_path, video_name),
                          batch_size= batch_size,
                          num_workers= num_workers,
                          gpu_params= gpu_params,
                          model= model,
                          compound_coef= compound_coef,
                          params= params,
                          thresholds= thresholds,
                          out_path= out_path,
                          tracker= tracker)
