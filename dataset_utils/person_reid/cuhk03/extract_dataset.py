from argparse import ArgumentParser
from pathlib import Path
import shutil


def main(img_dir: Path, gen_path: Path):
    for img_file in img_dir.glob("*.png"):
        pid, idk, idk = img_file.name.split("_")

        pid_path = gen_path / f"CUHK03_{pid}"
        pid_path.mkdir(parents=True, exist_ok=True)

        shutil.copy(img_file, pid_path / img_file.name)


if __name__ == "__main__":
    parser = ArgumentParser(description="Organize cuhk03 to directories format,"
                                        " a la face rec dataset style.")
    parser.add_argument("--img-dir", type=Path, required=True,
                        help="Path to the cuhk03 images")

    parser.add_argument("--output-path", type=Path, default="./generated/",
                        help="Where to put generated files")
    args = parser.parse_args()
    main(img_dir=args.img_dir,
         gen_path=args.output_path)
