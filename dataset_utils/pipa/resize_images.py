import argparse
import os
import shutil
from pathlib import Path

import cv2


def resize_all(folder_path: Path, image_size):
    for dir in folder_path.iterdir():
        if not dir.is_dir():
            continue
        for image_path in dir.iterdir():
            if not image_path.is_file():
                continue
            if ".DS_STORE" in str(image_path):
                continue

            img = cv2.imread(str(image_path))
            img = cv2.resize(img, tuple(image_size))
            cv2.imwrite(str(image_path), img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize all images")
    parser.add_argument('-f', '--folder_path', type=Path)
    parser.add_argument('-i', '--image_size', default=[160, 160],
                        type=int, nargs="+")
    args = parser.parse_args()
    resize_all(folder_path=args.folder_path,
               image_size=args.image_size)
