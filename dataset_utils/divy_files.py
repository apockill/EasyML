from argparse import ArgumentParser
from pathlib import Path
import shutil
import random


def main(input_dir: Path, output_dir: Path, num_move, filetype=None):
    if filetype is None:
        paths = list(Path(input_dir).iterdir())
    else:
        paths = list(Path(input_dir).glob("*" + filetype))
    random.shuffle(paths)

    for path in paths[:num_move]:
        shutil.move(path, output_dir / path.name)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Move a number of files from one directory to another."
                    " Useful when making a test set.")
    parser.add_argument("--input_dir", required=True,
                        help="Path to move files from")
    parser.add_argument("--output_dir", required=True,
                        help="Path to move files to")
    parser.add_argument("-n", "--num_move", required=True, type=int,
                        help="Percent in 0 to 100, float")
    parser.add_argument("-f", "--filetype", default=None,
                        help="Move only files of this file type")
    args = parser.parse_args()

    main(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        num_move=args.num_move,
        filetype=args.filetype)
