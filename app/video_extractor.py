import cv2
import os
import mediapipe as mp
import numpy as np
import csv

import unicodedata
import urllib.parse

from mp import MP_model
from landmark_visualization import draw_landmarks_on_image

NUM_LANDMARKS = 21
STATS = ["mean", "std"]

def convert_frame_to_mp_image(frame) -> mp.Image:
    # Convert frame to BGR for better recognition?
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Convert the frame to a MediaPipe Image object
    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_bgr
    )

def build_header() -> list:
    hands = ["l", "r"]
    coords = ["x", "y", "z"]

    header = []

    for hand in hands:
        for landmark_idx in range(NUM_LANDMARKS):
            for coord in coords:
                for stat in STATS:
                    header.append(f"{hand}_{coord}_{landmark_idx}_{stat}")

    return header

def build_row(results, label: str) -> list:
    """
    :param results: The collected results from a detection run on a video.
    :type results: list[HandLandmarkerResult]

    :param label: The corresponding label for the data in this video.
    :type label: str

    :return: A list with aggregated data for each coordinate of each landmark.
    :rtype: list[Any]
    """

    LEFT_SLOT = 0
    RIGHT_SLOT = 1
    FILL_VALUE = -11111111

    slots = [[], []]

    for result in results:
        # skip empty results
        if len(result.hand_world_landmarks) == 0:
            continue

        # assign landmarks in result to either left or right hand
        assignment = [None, None] 
        for i in range(len(result.hand_world_landmarks)):
            landmarks = result.hand_world_landmarks[i]
            handedness = result.handedness[i][0].category_name

            # decide preferred slot
            preferred_slot = LEFT_SLOT if handedness.lower() == "left" else RIGHT_SLOT

            if assignment[preferred_slot] is None:
                assignment[preferred_slot] = landmarks
            else:
                other_slot = RIGHT_SLOT if preferred_slot == LEFT_SLOT else LEFT_SLOT
                if assignment[other_slot] is None:
                    assignment[other_slot] = landmarks

        # collect results of left hands
        if assignment[LEFT_SLOT] is not None:
            slots[LEFT_SLOT].append(assignment[LEFT_SLOT])

        # collect results of right hands
        if assignment[RIGHT_SLOT] is not None:
            slots[RIGHT_SLOT].append(assignment[RIGHT_SLOT])

    # aggregate mean, min, max for each coordinate
    all_features = []
    for slot in slots:
        if len(slot) == 0:
            # fill with fill value if no hand was detected in the entire video
            all_features.extend([FILL_VALUE] * NUM_LANDMARKS * 3 * len(STATS))
            continue

        buckets = [{"x": [], "y": [], "z": []} for _ in range(NUM_LANDMARKS)]

        for landmarks in slot:
            for idx, landmark in enumerate(landmarks):
                buckets[idx]["x"].append(landmark.x)
                buckets[idx]["y"].append(landmark.y)
                buckets[idx]["z"].append(landmark.z)

        for bucket in buckets:
            for coord in ("x", "y", "z"):
                values = np.array(bucket[coord])
                # replace values.min() and .max() with values.std() for use with standard deviation. also adjust the STATS constant!
                all_features.extend([
                    values.mean(),
                    values.std()
                ])

    all_features.append(label)
    return all_features

def normalize_name(name: str) -> str:
    name = os.path.basename(name)
    name = urllib.parse.unquote(name)
    name = unicodedata.normalize("NFKC", name)
    name = (
        name.replace("•", "")
            .replace(" ", "_")
            .replace("/", "_")
    )
    return name.lower()

def build_video_lookup(csv_path: str) -> dict:
    lookup = {}

    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            video = row.get("videos")
            word = row.get("word")

            if not (video and word):
                continue

            key = normalize_name(video)

            if key not in lookup:
                lookup[key] = word

    return lookup

if __name__ == "__main__":
    files = os.scandir("input")
    time = 0 # continuously running index to satisfy mediapipes need for a timestamp

    model = MP_model("hand_landmarker.task")
    model.init_video()

    csv_file = open("table.csv", "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_header = build_header()
    csv_header.append("sign")
    csv_writer.writerow(csv_header)

    video_lookup = build_video_lookup("dataset.csv")

    for file in files:
        print("Processing " + file.name + "...")

        label = video_lookup.get(normalize_name(file.name))
        if label is None:
            print("Couldn't find a label for this video, skipping it...")
            continue

        results = []

        # Loop through each frame in the video
        cap = cv2.VideoCapture(file.path)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            mp_image = convert_frame_to_mp_image(frame)
            detection_result = model.landmarker.detect_for_video(mp_image, time)
            results.append(detection_result)
            # draw result on the original frame (consider using mp_image.numpy_view() for viewing the image mediapipe actually works with)
            annotated_image = draw_landmarks_on_image(frame, detection_result)

            cv2.imshow("Verification", annotated_image)
            cv2.waitKey(1) # opens the window and displays it for the given number of miliseconds

            time += 1

        cap.release()
        cv2.destroyAllWindows()

        csv_writer.writerow(build_row(results, label.lower()))

    csv_file.close()