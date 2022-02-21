# video-ocr

video-ocr is a command line tool and a python library that performs OCR on video frames, reducing the computational effort by choosing only frames that are different from their adjacent frames.


## Installation

1. video-ocr uses [tesserocr](https://github.com/sirfz/tesserocr), which in turn uses [tesseract](https://github.com/tesseract-ocr/tesseract) as the OCR engine. So, please follow the tesserocr installation instructions.
2. Install using pip: `pip install video-ocr`

## Getting started
### Command line usage
Run `video-ocr --help` to show the help text. Example usage is
```bash
video-ocr /path/to/video.mp4 --sample_rate=1 --debug_dir="path/to/debug/dir"
```
### Python usage
```python
import video_ocr
video_ocr.perform_video_ocr("/Users/veneetreddy/Downloads/texts.mp4", sample_rate=1)
```

