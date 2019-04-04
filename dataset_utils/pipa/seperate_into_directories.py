from argparse import ArgumentParser
from pathlib import Path
import shutil
from typing import List

import cv2

from easy_inference.models.object_detection import ObjectDetector
from easy_inference.bbox_utils import bb_intersection_over_union


def iter_data(labels_file: Path, data_dirs, detector: ObjectDetector):
    """Iterate over the crop of every identity in the dataset"""
    with open(labels_file, "r") as labels_file:
        lines = labels_file.readlines()

    def read_infer_and_crop(path, x, y, w, h):
        """Reads and crops the image, returns numpy array"""
        img = cv2.imread(str(path))
        objects = detector.predict([img])[0]
        people = [o for o in objects if o.name == "person"]

        if len(people) == 0:
            print("Skipped: No people found")
            return None

        face_rect = [x, y, x + w, x + h]
        people.sort(reverse=True,
                    key=lambda o: bb_intersection_over_union(face_rect, o.rect))

        best_fit = people[0]
        rect = best_fit.rect
        final_iou = bb_intersection_over_union(face_rect, rect)
        if final_iou < 0.1:
            print("Skipped: Not high enough IOU")
            return None

        cropped = img[rect[1]:rect[3], rect[0]:rect[2]]
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
                cropped = read_infer_and_crop(img_path, x, y, w, h)
                yield cropped, identity_id, file_name, data_dir
                break
        else:
            raise FileNotFoundError("Could not find image in either train or" +
                                    f" test directory! {file_name}")


def sort_imgs(data_dirs: List[Path], labels_file: Path,
              out_dir: Path, detector: ObjectDetector):
    data_iterator = iter_data(
        labels_file=labels_file,
        data_dirs=data_dirs,
        detector=detector)

    # Read, crop, and save all of the images
    for cropped_img, identity_id, file_name, data_dir in data_iterator:
        print("Saving person", identity_id)
        if cropped_img is None:
            # Person detector didn't find the person
            continue

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
    parser.add_argument("--model-file", type=Path, required=True,
                        help="Path to the *.pb model file")
    parser.add_argument("--labels-json", type=Path, required=True,
                        help="Path to the labels-json for the model")
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
    assert args.labels_json.is_file()
    assert args.model_file.is_file()

    object_detector = ObjectDetector.from_path(
        model_path=args.model_file,
        labels_path=args.labels_json)
    sort_imgs(
        detector=object_detector,
        data_dirs=args.data_dirs,
        labels_file=args.labels_file,
        out_dir=args.out_dir)


if __name__ == "__main__":
    main()
