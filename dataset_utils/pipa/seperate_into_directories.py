from argparse import ArgumentParser
from pathlib import Path
import shutil
from typing import List

import cv2


def iter_data(labels_file: Path, data_dirs):
    """Iterate over the crop of every identity in the dataset"""
    with open(labels_file, "r") as labels_file:
        lines = labels_file.readlines()

    def read_and_crop(path, x, y, w, h):
        """Reads and crops the image, returns numpy array"""
        img = cv2.imread(str(path))
        cropped = img[y: y + h, x:x + w]
        return cropped

    for line in lines:
        line = line.split()
        photoset_id, photo_id, x, y, w, h, identity_id, subset_id = line
        x, y, w, h = int(x), int(y), int(w), int(h)
        file_name = f"{photoset_id}_{photo_id}.jpg"

        # Find out where the image is
        for data_dir in data_dirs:
            img_path = data_dir / file_name
            if img_path.is_file():
                cropped = read_and_crop(img_path, x, y, w, h)
                yield cropped, identity_id, file_name, data_dir
                break
        else:
            raise FileNotFoundError("Could not find image in either train or" +
                                    f" test directory! {file_name}")


def sort_imgs(data_dirs: List[Path], labels_file: Path,
              out_dir: Path):
    data_iterator = iter_data(
        labels_file=labels_file,
        data_dirs=data_dirs
    )

    # Read, crop, and save all of the images
    for cropped_img, identity_id, file_name, data_dir in data_iterator:
        print("Saving person", identity_id)

        # Get/Make the save directory
        person_dir = out_dir / data_dir.name / str(identity_id)
        person_dir.mkdir(parents=True, exist_ok=True)
        save_path = person_dir / file_name
        assert not save_path.exists()

        # Save the image
        cv2.imwrite(str(save_path), cropped_img)


def main():
    parser = ArgumentParser(
        description="Sort all images into directories based on the person ID, "
                    "with the people cropped out")
    parser.add_argument("--data-dirs", type=Path, nargs="+",
                        help="Paths to the train, test, and val directories")
    parser.add_argument("--labels-file", type=Path,
                        help="Path to the index.txt with information about the "
                             "images")
    parser.add_argument("--out-dir", type=Path,
                        help="Path to the output directory to copy images to")
    args = parser.parse_args()

    assert all([d.is_dir() for d in args.data_dirs])
    assert args.labels_file.is_file()

    sort_imgs(
        data_dirs=args.data_dirs,
        labels_file=args.labels_file,
        out_dir=args.out_dir)


if __name__ == "__main__":
    main()
