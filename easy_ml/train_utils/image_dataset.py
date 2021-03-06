import random
import random
from pathlib import Path
from threading import Thread
from queue import Queue
from time import sleep

import cv2
import numpy as np


class Dataset:
    def __init__(self, img_dir, fraction_test_set, val_range=(0, 255),
                 dtype=np.uint8, load_resolution=None, max_size=None,
                 shuffle=False, processing_hook=None):
        """
        :param img_dir: A directory full of *.png images
        :param fraction_test_set: 0-1, the fraction of images to be put in the
        test set
        :param load_resolution: (width, height, ch) to load the images at.
        If None, it will use the original size
        :param processing_hook: This is a function that runs after loading,
        resizing, and color-converting an image.
        """
        assert 0 <= fraction_test_set < 1, \
            "The fraction test set must be between 0 and 1!"

        self.img_dir = img_dir
        self.fraction_test = fraction_test_set
        self.dtype = dtype
        self.val_range = val_range
        self.img_paths = list(Path(self.img_dir).glob("*.png")) + \
                         list(Path(self.img_dir).glob("*.jpg"))
        self.processing_hook=processing_hook
        if shuffle:
            random.shuffle(self.img_paths)
        if max_size is not None:
            self.img_paths = self.img_paths[:max_size]

        random.seed(0)
        random.shuffle(self.img_paths)

        if load_resolution is None:
            """Get a sample image in order to figure out the height and width 
            of the dataset samples"""

            sample_img = cv2.imread(str(self.img_paths[0]))
            assert sample_img is not None, "The Dataset object was unable to load an image!"
            shape = sample_img.shape
            self.height = shape[0]
            self.width = shape[1]
            self.channels = shape[2]
        else:
            self.height = load_resolution[0]
            self.width = load_resolution[1]
            self.channels = load_resolution[2]

    def __len__(self):
        return len(self.img_paths)

    def load(self, workers=8):
        """
        Return two arrays:
        x_train: [img_bgr, img_bgr, img_bgr]
        x_test: Same as x_train, but the images are for testing only.
        """
        num_test = len(self) * self.fraction_test

        x_test = []
        x_train = []

        work_queue = Queue()
        img_queue = Queue()

        # Create work
        for i, img_path in enumerate(self.img_paths):
            work_queue.put(img_path)

        # Start worker threads
        workers = [Thread(target=self._worker, args=(work_queue, img_queue))
                   for _ in range(workers)]
        for worker in workers:
            worker.start()

        # Wait for work to finish
        work_queue.join()

        # End worker threads
        for worker in workers:
            worker.join()

        # Get results
        while not img_queue.empty():
            img = img_queue.get()

            if len(x_test) < num_test:
                x_test.append(img)
            else:
                x_train.append(img)

        return np.asarray(x_train), np.asarray(x_test)

    def _load_and_preprocess(self, img_path):
        """Load an image and preprocess it according to spec"""
        img: np.ndarray = cv2.imread(str(img_path))

        # Adjust the image resolution
        if img.shape[:2] != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height))

        # Adjust the channel shape
        if img.shape[2] != self.channels:
            if img.shape[2] == 3:
                # Convert the image to gray, but increase dimensionality to
                # match the dimensions of a color image
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.expand_dims(img, axis=-1)
            elif img.shape[2] == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Run any custom image processing
        if self.processing_hook is not None:
            img = self.processing_hook(img)

        # Adjust the image to be between two specific values
        img = img.astype(dtype=np.float32)  # Temporarily turn to float
        img /= 255
        img = img * (self.val_range[1] - self.val_range[0])
        img += self.val_range[0]
        img = np.clip(img, *self.val_range)

        # Adjust the images type
        img = img.astype(self.dtype)

        return img

    def _worker(self, work_queue, img_queue):
        while not work_queue.empty():
            img_path = work_queue.get()
            img = self._load_and_preprocess(img_path)
            img_queue.put(img)
            work_queue.task_done()