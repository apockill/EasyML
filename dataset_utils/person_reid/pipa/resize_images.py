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

            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Removing {image_path}, not a valid image!")
                os.remove(image_path)
                continue

            if list(img.shape)[:2] == image_size:
                print("Skipping, already correct size")
                continue

            img = cv2.resize(img, tuple(image_size))
            cv2.imwrite(str(image_path), img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize all images")
    parser.add_argument('-f', '--folder_path', type=Path, required=True)
    parser.add_argument('-i', '--image_size', default=[180, 180],
                        type=int, nargs="+")
    args = parser.parse_args()
    resize_all(folder_path=args.folder_path,
               image_size=args.image_size)
