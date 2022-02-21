import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="video-ocr",                     # This is the name of the package
    version="0.0.1",                        # The initial release version
    author="Veneet Reddy",                     # Full name of the author
    description="Package to run OCR on videos",
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),    # List of all python modules to be installed
    entry_points = {
                'console_scripts': ['video-ocr=video_ocr:main'],
                    }
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 2 - Pre-Alpha"
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],                                      # Information to filter the project on PyPi website
    python_requires='>=3.6',                # Minimum version requirement of the package
    py_modules=["video_ocr"],             # Name of the python package
    install_requires=[
        "tesserocr~=2.5.2",
        "scipy~=1.7.1",
        "opencv~=4.5.2",
        "numpy~=1.21.1",
        "tqdm~=4.62.3",
        "click~=8.0.3",
    ],
    extras_require={
        "dev": [
            "flake8~=4.0.1",
            "black~=19.3b0",
        ]
    }
)
