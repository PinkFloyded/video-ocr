import click
import cv2 as cv

import scipy.fft
from contextlib import contextmanager
import numpy as np
from itertools import tee
import tesserocr
from PIL import Image
from multiprocessing.pool import ThreadPool
import multiprocessing
from config import error_log, info_log
import tqdm

class NoOpProgressBar:
    def update(self):
        pass
    def total(self, n):
        pass


pbar = NoOpProgressBar()


def phash_faster(image, hash_size=8, highfreq_factor=4):
    img_size = hash_size * highfreq_factor
    image = cv.resize(image, (img_size, img_size), interpolation=cv.INTER_LINEAR)
    dct = scipy.fft.dct(scipy.fft.dct(image, axis=0), axis=1)
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq)
    diff = dctlowfreq > med
    return diff

class Frame:
    def __init__(self, frame_number, image, millis):
        self.frame_number = frame_number
        self.image = image
        self. millis = millis



@contextmanager
def open_cv_video(filepath):
    cap = cv.VideoCapture(filepath)
    try:
        yield cap
    finally:
        cap.release()

def get_frames(video_capture):
    fps = int(video_capture.get(cv.CAP_PROP_FPS))
    pbar.total = (video_capture.get(cv.CAP_PROP_FRAME_COUNT) // fps) - 1
    frame_number = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_number += 1
        if frame_number % fps != 0:
            continue
        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        yield Frame(frame_number, frame, video_capture.get(cv.CAP_PROP_POS_MSEC))


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def are_similar_frame(f1, f2):
    diff = np.count_nonzero(phash_faster(f1.image) != phash_faster(f2.image))
    return diff <= 15

def filter_redundant_frames(frames):
    for f1, f2 in pairwise(frames):
        if not are_similar_frame(f1, f2):
            yield f1
        else:
            pbar.update()

def ocr(frame):
    pil_image = Image.fromarray(frame.image)
    text = tesserocr.image_to_text(pil_image)
    frame.text = text
    pbar.update()
    return frame

def parallel_ocr(frames):
    with ThreadPool(multiprocessing.cpu_count()) as pool:
        return pool.map(ocr, frames, chunksize=multiprocessing.cpu_count())
        
def perform_video_ocr(filepath, fps=None, sample_rate=1, debug_dir=""):
    c = 0
    frames = []
    with open_cv_video(filepath) as cap:
        frames = parallel_ocr(filter_redundant_frames(get_frames(cap)))
        for frame in frames:
            cv.imwrite(f"local/imgs/{frame.frame_number}.jpg", frame.image)
            with open(f"local/imgs/{frame.frame_number}.txt", "w") as f:
                f.write(frame.text)
            

@click.command()
@click.argument('filepath', type=click.Path(exists=True, readable=True))
@click.option('--fps', type=int)
@click.option('--sample_rate', type=int)
@click.option('--debug_dir', type=click.Path(exists=True, writable=True, file_okay=False, dir_okay=True))
def main(filepath, fps, sample_rate, debug_dir):
    global pbar
    with tqdm.tqdm() as progress_bar:
        pbar = progress_bar
        perform_video_ocr(filepath)

if __name__ == '__main__':
    main()
