# EfficientDet Pytorch

The pytorch re-implement of the official [EfficientDet](https://github.com/google/automl/tree/master/efficientdet) with SOTA performance in real time, original paper link: https://arxiv.org/abs/1911.09070

# Table of Contents

* [COCO Performance](#COCO Performance)
* [Training](#Training)
  - [Synthetic data gen](./synthetic_signs/README.md)
* [Evaluation](#Evaluate model performance)
* [Inference](#Inference)



# COCO Performance

## Pretrained weights and benchmark

The performance is very close to the paper's, it is still SOTA. 

The speed/FPS test includes the time of post-processing with no jit/data precision trick.

| coefficient | pth_download | GPU Mem(MB) | FPS | Extreme FPS (Batchsize 32) | mAP 0.5:0.95(this repo) | mAP 0.5:0.95(paper) |
| :-----: | :-----: | :------: | :------: | :------: | :-----: | :-----: |
| D0 | [efficientdet-d0.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d0.pth) | 1049 | 36.20 | 163.14 | 32.6 | 33.8
| D1 | [efficientdet-d1.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d1.pth) | 1159 | 29.69 | 63.08 | 38.2 | 39.6
| D2 | [efficientdet-d2.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d2.pth) | 1321 | 26.50 | 40.99 | 41.5 | 43.0
| D3 | [efficientdet-d3.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d3.pth) | 1647 | 22.73 | - | 44.9 | 45.8
| D4 | [efficientdet-d4.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d4.pth) | 1903 | 14.75 | - | 48.1 | 49.4
| D5 | [efficientdet-d5.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d5.pth) | 2255 | 7.11 | - | 49.5 | 50.7
| D6 | [efficientdet-d6.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d6.pth) | 2985 | 5.30 | - | 50.1 | 51.7
| D7 | [efficientdet-d7.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d7.pth) | 3819 | 3.73 | - | 50.7 | 52.2

## Speed Test

This pure-pytorch implement is up to 2 times faster than the official Tensorflow version without any trick.

Recorded on 2020-04-26, 

official git version: https://github.com/google/automl/commit/006668f2af1744de0357ca3d400527feaa73c122

| coefficient | FPS(this repo, tested on RTX2080Ti) | FPS(official, tested on T4) |  Ratio |
| :------: | :------: | :------: | :-----: |
| D0 | 36.20 | 42.1 | 0.86X |
| D1 | 29.69 | 27.7 | 1.07X |
| D2 | 26.50 | 19.7 | 1.35X |
| D3 | 22.73 | 11.8 | 1.93X |
| D4 | 14.75 | 7.1 | 2.08X |
| D5 | 7.11 | 3.6 | 1.98X |
| D6 | 5.30 | 2.6 | 2.03X |
| D7 | 3.73 | - | - |


Test method (this repo):

Run this test on 2080Ti, Ubuntu 19.10 x64.
1. Prepare a image tensor with the same content, size (1,3,512,512)-pytorch.
2. Initiate everything by inferring once.
3. Run 10 times with batchsize 1 and calculate the average time, including post-processing and visualization, to make the test more practical.



# Training

## 1. Manual set project's specific parameters

```yaml
# create a yml file {your_project_name}.yml under 'projects'folder 
# modify it following 'coco.yml'
 
# for example
project_name: coco
train_set: train2017
val_set: val2017
num_gpus: 4  # 0 means using cpu, 1-N means using gpus 

# mean and std in RGB order, actually this part should remain unchanged as long as your dataset is similar to coco.
mean: [0.485, 0.456, 0.406]
std: [0.229, 0.224, 0.225]

# this is coco anchors, change it if necessary
anchors_scales: '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
anchors_ratios: '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'

# objects from all labels from your dataset with the order from your annotations.
# its index must match your dataset's category_id.
# category_id is one_indexed,
# for example, index of 'car' here is 2, while category_id of is 3
obj_list: ['person', 'bicycle', 'car', ...]
```
## 2. Prepare your dataset

    # your dataset structure should be like this
    datasets/
        -your_project_name/
            -train_set_name/
                -*.jpg
            -val_set_name/
                -*.jpg
            -annotations
                -instances_{train_set_name}.json
                -instances_{val_set_name}.json
    
    # for example, coco2017
    datasets/
        -coco2017/
            -train2017/
                -000000000001.jpg
                -000000000002.jpg
                -000000000003.jpg
            -val2017/
                -000000000004.jpg
                -000000000005.jpg
                -000000000006.jpg
            -annotations
                -instances_train2017.json
                -instances_val2017.json

Use a `dataset_to_coco.py` to convert to COCO format.

For GTSDB:

1. [Download](https://sid.erda.dk/public/archives/ff17dc924eba88d5d01a807357d6614c/published-archive.html) FullIJCNN2013 set

2. Copy first 599 files to train_set_name/ folder and last 300 files to val_set_name/

3. Copy gt.txt to annotations/ folder

4. Run `dataset_to_coco.py` with `--dataset_type gtsdb ` (The project file must be filled)

For synthetic signs dataset:

1. [Generate](./synthetic_signs/README.md)  Train and Test sets
2. Run `dataset_to_coco.py` with `--dataset_type synth_signs` (The project file must be filled)
3. Remove `multiclass.csv`, `binary.csv` from sets folder. Move images from `imgs` to sets folder

## 3. Train a custom dataset with pretrained weights

```bash
# train efficientdet-d2 on a custom dataset with pretrained weights
# with batchsize 8 and learning rate 1e-5 for 10 epoches

python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-5 --num_epochs 10 \
 --load_weights /path/to/your/weights/efficientdet-d2.pth

# with a coco-pretrained, you can even freeze the backbone and train heads only
# to speed up training and help convergence.

python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-5 --num_epochs 10 \
 --load_weights /path/to/your/weights/efficientdet-d2.pth \
 --head_only True
```

## 4. Early stopping a training session

```bash
# while training, press Ctrl+c, the program will catch KeyboardInterrupt
# and stop training, save current checkpoint.
```

## 5. Resume training

```bash
# let says you started a training session like this.

python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-5 \
 --load_weights /path/to/your/weights/efficientdet-d2.pth \
 --head_only True
 
# then you stopped it with a Ctrl+c, it exited with a checkpoint

# now you want to resume training from the last checkpoint
# simply set load_weights to 'last'

python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-5 \
 --load_weights last \
 --head_only True
```

## 6. Tensorboard

```bash
# use tensorboard to look at the train and val plots

tensorboard --logdir logs/your_project_name/tensorboard/ --port 6006
```
## 7. Debug training (optional)

```bash
# when you get bad result, you need to debug the training result.
python train.py -c 2 -p your_project_name --batch_size 8 --lr 1e-5 --debug True

# then checkout test/ folder, there you can visualize the predicted boxes during training
# don't panic if you see countless of error boxes, it happens when the training is at early stage.
# But if you still can't see a normal box after several epoches, not even one in all image,
# then it's possible that either the anchors config is inappropriate or the ground truth is corrupted.
```

## 8. Docker
```bash
# you can train model with docker
#first build image
docker build -f Dockerfile.train -t tsd/train .

#input, output and weights folders should be mounted as volume
	#-v C:\dataset:/app/input/ \    - folder with datasets
	#-v C:\output:/app/output/ \    - folder for logs and checkpoint weights
	#-v C:\weights:/app/weights/    - folder with weights and project files
##then run with arguments in yaml project file
  --compound_coef COMPOUND_COEF
                        coefficients of efficientdet
  --num_workers NUM_WORKERS
                        num_workers of dataloader
  --batch_size BATCH_SIZE
                        The number of images per batch among all devices
  --head_only HEAD_ONLY
                        whether finetunes only the regressor and the
                        classifier, useful in early stage convergence or
                        small/easy dataset
  --lr LR
  --optim OPTIM         select optimizer for training, suggest using 'admaw'
                        until the very final stage then switch to 'sgd'
  --num_epochs NUM_EPOCHS
  --val_interval VAL_INTERVAL
                        Number of epoches between valing phases
  --save_interval SAVE_INTERVAL
                        Number of steps between saving
  --es_min_delta ES_MIN_DELTA
                        Early stopping's parameter: minimum change loss to
                        qualify as an improvement
  --es_patience ES_PATIENCE
                        Early stopping's parameter: number of epochs with no
                        improvement after which training will be stopped. Set
                        to 0 to disable this technique.
  --load_weights LOAD_WEIGHTS
                        whether to load weights from a checkpoint, set None to
                        initialize, set 'last' to load last checkpoint
  --batch_multiplier BATCH_MULTIPLIER
                        Increasing Mini-batch Size without Increasing Memory
  
#example
docker run --gpus=all -it --rm -v C:\EfficientDet-Pytorch\datasets\:/app/input/ \
-v C:\EfficientDet-Pytorch\logs:/app/output/ \
-v C:\EfficientDet-Pytorch\logs\synth_signs:/app/weights/ \
tsd/train 
#project.yml
	compound_coef: 3,
	load_weights: best_mapillary_efficientdet-d3_0_1.pth,
	batch_size: 1, 
	num_workers: 0,
	head_only: True
```


# Evaluate model performance

```bash
# eval on your_project, efficientdet-d3

python coco_eval.py -p your_project_name -c 3 \
 -w /path/to/your/weights
```

# Inference

```bash
#detect traffic signs on a video

python inference.py -p your_project_name -c 3 -i path/to/video/or/folder -o path/to/output \
 -w /path/to/your/weights
```
For better performance of the detector, use tracking `--tracking True`. For more info see [SORT](https://github.com/abewley/sort).

```bash
#visualize bboxes
python visualize_inference.py -p your_project_name -c 3 --video_path path/to/video/ --detection_path path/to/predicted/file \
 --destination path/to/output.avi

#put class image instead of the classname
python visualize_inference.py -p your_project_name -c 3 --video_path path/to/video/ --detection_path path/to/predicted/file \
 --destination path/to/output.avi --template-dir path/to/folder/with/images
```
### Docker

```bash
# you can use inference with docker
#first build image
docker build -f Dockerfile.inference -t tsd/inference .

#input, output and weights folders should be mounted as volume
	#-v C:\dataset:/app/input/ \    - contains videos for detection
	#-v C:\output:/app/output/ \    - folder for json prediction
	#-v C:\weights:/app/weights/    - folder with weights and project files
#then run with arguments in yaml project file
  --compound_coef COMPOUND_COEF
                        coefficients of efficientdet
  --weights WEIGHTS
                        /path/to/weights
  --nms_threshold NMS_THRESHOLD
                        nms threshold, don't change it if not for testing
                        purposes
  --threshold THRESHOLD
                        threshold of classification, don't change it if not
                        for testing purposes
  --cuda CUDA
  --device DEVICE
  --float16 FLOAT16
  --batch_size BATCH_SIZE
                        The number of images per batch among all devices
  --num_workers NUM_WORKERS
                        num_workers of dataloader
  --tracking TRACKING   Apply tracker
  --max_age MAX_AGE     Max frames to keep object id without bbox
  --min_hits MIN_HITS   Min frames with object set id
 
#example
docker run --gpus=all -it --rm -v C:\EfficientDet-Pytorch\input\:/app/input/ \
-v C:\EfficientDet-Pytorch\output:/app/output/ \
-v C:\EfficientDet-Pytorch\logs\synth_signs:/app/weights/ \
tsd/inference 
#project.yml
	compound_coef: 3,
	weights: "best_mapillary_efficientdet-d3_0_1.pth",
	batch_size: 1,
	num_workers: 0

```
## References

- [zylo117/Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)
- [abewley/sort](https://github.com/abewley/sort)
- [moabitcoin/signfeld](https://github.com/moabitcoin/signfeld)

- [google/automl](https://github.com/google/automl)
- [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
- [signatrix/efficientdet](https://github.com/signatrix/efficientdet)
- [vacancy/Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)
