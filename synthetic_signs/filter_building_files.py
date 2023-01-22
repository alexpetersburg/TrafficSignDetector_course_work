"""
Use this script to To filter images from traffic signs.
Usage:
  Example:  python filter_building_files.py  --building_path external/lists/Building.list
                                             --sign_path external/lists/Traffic_sign.list
                                             --light_path external/lists/Traffic_light.list
                                             --out_path external/lists/Building_without_signs.list
  For help: python filter_building_files.py -h

"""
import argparse
import os
import pathlib
import itertools


def parse_args():
    """Parse command line arguments.

    Returns:
    output (argparse.Namespace): Parsed command line arguments. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--building_path",
                      type=pathlib.Path,
                      default= pathlib.Path('external/lists/Building.list'),
                      help="Path to the Building.list")
    parser.add_argument("--sign_path",
                      type=pathlib.Path,
                      default= pathlib.Path('external/lists/Traffic_sign.list'),
                      help="Path to the Traffic_sign.list")
    parser.add_argument("--light_path",
                      type=pathlib.Path,
                      default= pathlib.Path('external/lists/Traffic_light.list'),
                      help="Path to the Traffic_light.list")
    parser.add_argument("--out_path",
                      type=pathlib.Path,
                      default= pathlib.Path('external/lists/Building_without_signs.list'),
                      help="Out path")
    return parser.parse_args()

def main(args):
    assert os.path.exists(args.building_path), "Building file does not exist"
    with args.building_path.open() as file_handle:
        building_filenames = file_handle.readlines()
    building_filenames = [filename.rstrip() for filename in building_filenames]
    assert os.path.exists(args.sign_path), "Sign file does not exist"
    with args.sign_path.open() as file_handle:
        sign_filenames = file_handle.readlines()
    sign_filenames = [pathlib.Path(filename.rstrip()).name for filename in sign_filenames]
    assert os.path.exists(args.light_path), "Light file does not exist"
    with args.light_path.open() as file_handle:
        light_filenames = file_handle.readlines()
    light_filenames = [pathlib.Path(filename.rstrip()).name for filename in light_filenames]
    blacklist = list(itertools.chain(sign_filenames, light_filenames))
    filtered_building = [filename for filename in building_filenames if pathlib.Path(filename).name not in blacklist]
    with args.out_path.open('w') as file_handle:
        file_handle.write("\n".join(filtered_building))
    print("Success")

if __name__ == "__main__":
  main(parse_args())
