from argparse import ArgumentParser
from pathlib import Path
from threading import Thread
from queue import Queue
from time import sleep

from quickdraw import QuickDrawData, QuickDrawDataGroup


def worker_thread(work_queue: Queue, save_img_dir: str, img_format: str):
    while True:
        name, index, drawing = work_queue.get()

        save_loc = Path(save_img_dir) / name
        save_loc.mkdir(parents=True, exist_ok=True)
        save_loc = save_loc / (name + "_" + str(index) + img_format)

        if save_loc.is_file():
            continue

        image = drawing.get_image(
            bg_color=(255, 255, 255),
            stroke_color=(0, 0, 0),
            stroke_width=2)

        image.save(save_loc)
        work_queue.task_done()


def main(cache_dir, save_img_dir, img_format, num_workers, categories,
         drawing_recognized):
    work = Queue()
    # Start worker threads
    for _ in range(num_workers):
        w = Thread(target=worker_thread,
                   args=(work, save_img_dir, img_format),
                   daemon=True)
        w.start()

    # Start feeding in work
    for name in categories:
        group = QuickDrawDataGroup(name=name,
                                   recognized=drawing_recognized,
                                   cache_dir=cache_dir,
                                   max_drawings=float('inf'))
        for i, drawing in enumerate(group.drawings):
            # Wait until some work gets done before continuing
            while work.qsize() > 10000:
                sleep(0.1)
            work.put((name, i, drawing))

    # Wait for work to finish
    work.join()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--cache_dir", help="Path to quickdraw cache dir")
    parser.add_argument("--save_img_dir",
                        help="Path to render images in sorted dirs")
    parser.add_argument("--image_format", default=".jpg",
                        help="Image format: .jpg or .png or .gif")
    parser.add_argument("--num_workers", type=int, default=24,
                        help="Number of worker threads")
    parser.add_argument("--categories", type=str, nargs="+", required=True,
                        help="List of categories to render")
    parser.add_argument("--recognized", type=str, nargs="+", default=None,
                        help="Render only recognized or unrecognized drawings.")
    args = parser.parse_args()

    main(cache_dir=args.cache_dir,
         save_img_dir=args.save_img_dir,
         img_format=args.image_format,
         num_workers=args.num_workers,
         categories=args.categories,
         drawing_recognized=args.recognized)

