import click
import cv2 as cv

import scipy.fft
import numpy as np

def phash_faster(image, hash_size=8, highfreq_factor=4):
    if hash_size < 2:
        raise ValueError("Hash size must be greater than or equal to 2")

    img_size = hash_size * highfreq_factor
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    image = cv.resize(image, (img_size, img_size), interpolation=cv.INTER_LINEAR)
    dct = scipy.fft.dct(scipy.fft.dct(image, axis=0), axis=1)
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq)
    diff = dctlowfreq > med
    return diff
        
def perform_video_ocr(filename):
    cap = cv.VideoCapture(filename)
    c = 0
    hashes = []
    while cap.isOpened():
        c += 1
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        if c % 120 == 0:
            hashes.append(phash_faster(frame))
        if c > 1000:
            break
    cap.release()
    print(hashes)


# from contextlib import contextmanager

# @contextmanager
# def managed_resource(*args, **kwds):
#     # Code to acquire resource, e.g.:
#     resource = acquire_resource(*args, **kwds)
#     try:
#         yield resource
#     finally:
#         # Code to release resource, e.g.:
#         release_resource(resource)


@click.command()
@click.argument('filepath', type=click.Path(exists=True))
def main(filepath):
    perform_video_ocr(filepath)

if __name__ == '__main__':
    main()
