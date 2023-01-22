"""
Use this script to visualize the inference results.
Input
-----
    video path
    prediction path
Output
-----
    video with bboxes
Usage:
  Example:  python vizualize_inference.py -p synth_signs --video_path input\video.mp4
                                                         --detection_path output\video.json"
                                                         --destination output\video.avi"
  For help: python vizualize_inference.py -h

"""

import argparse
import os
import pathlib
import json
import yaml

import cv2
import numpy as np
import torch
import torchvision
from tqdm import tqdm
import sys
sys.path.append(os.path.join(sys.path[0], "synthetic_signs/"))
from synthetic_signs.utils import clip_image_at_border, remove_padding
from synthetic_signs.blending import paste_to


DETECTION_COLOR_BGR = 0, 0, 255
LINE_THICKNESS = 2
TEXT_THICKNESS = 2


def draw_detection_box(image, x0, y0, x1, y1):
    pt1 = int(x0), int(y0)
    pt2 = int(x1), int(y1)
    cv2.rectangle(image, pt1, pt2, DETECTION_COLOR_BGR, LINE_THICKNESS)


def annotate_category_name(image, category, x0, y0, x1, y1):
    text_size = cv2.getTextSize(category, cv2.FONT_HERSHEY_PLAIN, 1,
                              TEXT_THICKNESS)

    pt1 = int(x0), int(y0)
    pt2 = int(x1), int(y1)
    pt1 = pt1[0] - 10 - text_size[0][0], pt1[1]
    pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + text_size[0][1]
    cv2.rectangle(image, pt1, pt2, DETECTION_COLOR_BGR, -1)

    center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
    cv2.putText(image, category, center, 1, cv2.FONT_HERSHEY_PLAIN,
              (255, 255, 255), TEXT_THICKNESS)


def annotate_template_image(image, template_bgra, x0, y0, x1, y1):
    scale_factor = float(y1 - y0) / template_bgra.shape[0]
    template_bgra = cv2.resize(template_bgra, (0, 0),
                             fx=scale_factor,
                             fy=scale_factor)

    offset = int(y0), int(x0 - template_bgra.shape[1])
    template_bgra, offset = clip_image_at_border(
        template_bgra, image.shape[:2], offset)

    try:

        image[:, :, :] = paste_to(template_bgra[:, :, :3],
                                    template_bgra[:, :, 3],
                                    image, offset)
    except Exception as err:
        return


def main(args):
    # Load templates into memory.
    name_to_image = dict()
    params = yaml.safe_load(open(f'projects/{args.project}.yml'))
    if args.template_dir is not None:
        for filepath in args.template_dir.iterdir():
            try:
                image_bgra = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
                assert image_bgra.ndim == 3 and image_bgra.shape[-1] == 4
                image_bgra, _ = remove_padding(
                image_bgra, image_bgra[:, :, 3])

                name, _ = os.path.splitext(filepath.name)
                name_to_image[name] = image_bgra
            except Exception as err:
                print('Not an image template {}'.format(filepath))

    # Process each frame in video.
    cap = cv2.VideoCapture(args.video_path)
    progress_bar = tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    out_cap = cv2.VideoWriter(args.destination, fourcc, fps, (width, height))
    with open(args.detection_path, 'r') as config_json:
        detections = json.load(config_json)
    for frame_num in progress_bar:
        ret, frame = cap.read()
        detection = detections[frame_num]['detection']
        if len(detection)>0:

            boxes = np.array([s['bbox'] for s in detection])
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]
            scores = [s['score'] for s in detection]
            categories = [params['obj_list'][s['category_id']-1] for s in detection]

            for (x0, y0, x1, y1), score, category in zip(boxes, scores, categories):
                draw_detection_box(frame, x0, y0, x1, y1)
                template_bgra = name_to_image.get(category, None)
                if template_bgra is None:
                    annotate_category_name(frame, category, x0, y0, x1, y1)
                else:
                    annotate_template_image(frame, template_bgra, x0, y0, x1, y1)
            out_cap.write(frame)
        else:
            out_cap.write(frame)



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    parser.add_argument("--video_path", required=True, type=str, help="Path to video")
    parser.add_argument("--detection_path",
                        required=True,
                        type=pathlib.Path,
                        help="Path to detection output/ground truths.")
    parser.add_argument("--destination",
                        required=True,
                        type=str,
                        help="Visualization is stored in this directory.")
    parser.add_argument("--template-dir",
                        type=pathlib.Path,
                        help="Path to templates")
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())
