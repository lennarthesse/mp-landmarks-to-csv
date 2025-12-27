# mp-landmarks-to-csv

This project uses a [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) landmark detection model to analyze the hands in a set of input images. The extracted landmark coordinates are then written into a `.csv` file along with the associated label. The label currently only supports a single letter from the alphabet which is provided by the first character of the filename.

At the moment we are using this script to get structured and labeled landmark data which is then used to train a tabular classification model.

## Installation

1. Create a virtual environment (optional):
    - `python3 -m venv .env`
    - `source .env/bin/activate`

2. Install dependencies:
    - `pip install -r requirements.txt`

## Preparing the input data

> **Important:** Currently the script only works with **.jpg** images!

To extract the hand landmarks from an image, the hand should be fully visible. The filename **must** start with a letter specifying the label associated with the image or else the file will be skipped.

## Running the script

1. Put the images you want to extract landmarks from into the `input` folder.

2. Navigate to the app folder and run the main script:
    - `cd app`
    - `python landmark-extractor.py`

3. Enter the name of the input directory and the name of the dataset you are creating. The latter will also be the name of the output directory.

MediaPipe will now analyze the images and the script will save a `.csv` file along with the analyzed images to the output directory.

## Example

`a.jpg` is used as an input image and put into the input folder:

![Sign of the letter A in german sign language](assets/a.jpg)

The script processes it and writes the following into the specified output directory:

1. `[output].csv`:

    | 0_x | 0_y | 0_z | ... | 20_x | 20_y | 20_z | sign | hash |
    | :-: | :-: | :-: | :-: | :--: | :--: | :--: | :--: | :--: |
    | 0.3252 | 0.7996 | -0.0000 | ... | 0.3047 | 0.6118 | -0.0829 | a | 6.58763415117154e+018 |

2. `a.jpg`:

    ![Input image with overlayed landmarks](assets/a-overlay.jpg)

The table contains the coordinates of each landmark along with the label (sign) and hash of the image. The image with the overlay serves to validate the MediaPipe output.