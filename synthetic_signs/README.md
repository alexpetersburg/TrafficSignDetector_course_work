## Neutral images
For the next steps we'd need images which do not contain any known traffic signs. We leverage [OpenimagesV5](https://storage.googleapis.com/openimages/web/index.html) and build a neutral image set by querying for [`Building`](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F0cgh4) and filtering out images containing [`traffic signs`](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F01mqdt), referred below as `Building_without_signs.list`. Please download the following file(s) and place them under `synthetic_signs/external/lists`:
- [train-annotations-bbox.csv](https://datasets.figure-eight.com/figure_eight_datasets/open-images/train-annotations-bbox.csv)
- [validation-annotations-bbox.csv](https://datasets.figure-eight.com/figure_eight_datasets/open-images/validation-annotations-bbox.csv)
- [test-annotations-bbox.csv](https://datasets.figure-eight.com/figure_eight_datasets/open-images/test-annotations-bbox.csv)

### Neutral images list gen

```
python synthetic_signs/download-openimages-v5.py --help
Download Class specific images from OpenImagesV5

optional arguments:
  -h, --help           show this help message and exit
  --mode MODE          Dataset category - train, validation or test
  --classes CLASSES    Names of object classes to be downloaded
  --nthreads NTHREADS  Number of threads to use
  --dest DEST          Destination directory
  --csvs CSVS          CSV file(s) directory
  --limit LIMIT        Cap downloaded files to limit
```
Sample commands
```
python synthetic_signs/download-openimages-v5.py --classes 'Building,Traffic_sign,Traffic_light' --mode train --dest synth_template/Train --csvs synthetic_signs/external/lists

python synthetic_signs/download-openimages-v5.py --classes 'Building,Traffic_sign,Traffic_light' --mode test --dest synth_template/Test --csvs synthetic_signs/external/lists
```
#### **All the steps below must be done first for the Train set, then for the Test set**

Build class list and filter image set 

```
find path_to_openimages_v5/Building -type f -name '*.jpg' > synthetic_signs/external/lists/Building.list
find path_to_openimages_v5/Traffic_sign -type f -name '*.jpg' > synthetic_signs/external/lists/Traffic_sign.list
find path_to_openimages_v5/Traffic_light -type f -name '*.jpg' > synthetic_signs/external/lists/Traffic_light.list
```

Filtering for outdoor images with no labeled signs.

```
#if you don't change synthetic_signs/external/lists/ path, use default settings
python synthetic_signs\filter_building_files.py -h
usage: filter_building_files.py [-h] [--building_path BUILDING_PATH]
                                [--sign_path SIGN_PATH]
                                [--light_path LIGHT_PATH]
                                [--out_path OUT_PATH]

optional arguments:
  -h, --help                      show this help message and exit
  --building_path BUILDING_PATH   Path to the Building.list
  --sign_path SIGN_PATH           Path to the Traffic_sign.list
  --light_path LIGHT_PATH         Path to the Traffic_light.list
  --out_path OUT_PATH             Out path
```

## Traffic sign templates

A subset of the near complete [list](https://de.wikipedia.org/wiki/Bildtafel_der_Verkehrszeichen_in_der_Bundesrepublik_Deutschland_seit_2017) of German traffic sign(s) is of interest to us. More specifically [these](https://github.com/moabitcoin/Signfeld/tree/master/synthetic_signs/templates). These signs form a subset (signs of interest) which formalise [`turn-restrictions`](https://wiki.openstreetmap.org/wiki/Relation:restriction) in OSM. Using the templates for the signs of interest we build a synthetic training set following the idea presented in [IJCN2019](https://github.com/LCAD-UFES/publications-tabelini-ijcnn-2019) and [CVPR 2016](https://github.com/ankush-me/SynthText).

## Generate synthetic data

Use `dataset_generator.py` to generate a synthetic sign dataset:
```
optional arguments:
  -h, --help            show this help message and exit
  --backgrounds BACKGROUNDS
                        Path to the directory containing background images to
                        be used (or file list).
  --templates-path TEMPLATES_PATH
                        Path (or file list) of templates.
  --augmentations AUGMENTATIONS
                        Path to augmentation configuration file.
  --distractors-path DISTRACTORS_PATH
                        Path (or file list) of distractors.
  --random-distractors RANDOM_DISTRACTORS
                        Generate this many random distractors for each
                        template.
  --out-path OUT_PATH   Path to the directory to save the generated images to.
  --max-images MAX_IMAGES
                        Number of images to be generated.
  --n JOBS              Maximum number of parallel processes.
  --max-template-size MAX_TEMPLATE_SIZE
                        Maximum template size.
  --min-template-size MIN_TEMPLATE_SIZE
                        Minimum template size.
  --background-size BACKGROUND_SIZE
                        If not None (or empty string), image shape
                        'height, width'
```
The following example generates a dataset of 2M images. The file `augmentations.yaml` specifies augmentation parameters (geometric template distortion, blending methods, etc.). Refer to the documentation of `generate_task_args()` in `dataset_generator.py` for a parameter description.

```
python dataset_generator.py --backgrounds=synthetic_signs/external/lists/Building_without_signs.list \
                            --templates-path=synthetic_signs/templates \
                            --out-path=dataset/synthetic-signs/Train \
                            --n=16 \
                            --max-images=200000 \
                            --augmentations=synthetic_signs/external/augmentations.yaml
```
