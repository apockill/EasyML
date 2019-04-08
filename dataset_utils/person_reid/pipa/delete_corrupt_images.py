import argparse
import os
import shutil

import cv2


def find_corrupt(folder_path):
    data_dir = folder_path
    flds = os.listdir(data_dir)

    for fld in flds:
        try:
            sub_flds = os.listdir(data_dir + '/' + fld)
        except NotADirectoryError:
            print(f"Skipped {fld}")
            continue

        for i in sub_flds:
            i_path = data_dir + '/' + fld + '/' + i
            img = cv2.imread(i_path)

            if img is None:
                print("Deleting", data_dir + '/' + fld)
                shutil.rmtree(data_dir + '/' + fld)
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     "Delete folders that have corrupt images")
    parser.add_argument('-f', '--folder_path', type=str)
    args = parser.parse_args()
    find_corrupt(str(args.folder_path))
