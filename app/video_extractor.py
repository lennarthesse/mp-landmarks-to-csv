import cv2
import os
import mediapipe as mp

import csv
from collections import deque

from mp import MP_model
from landmark_visualization import draw_landmarks_on_image

def convert_frame_to_mp_image(frame) -> mp.Image:
    # Convert frame to BGR for better recognition?
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Convert the frame to a MediaPipe Image object
    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=frame_bgr
    )

def build_frame_row(detection_result):

    LEFT_SLOT = 0
    RIGHT_SLOT = 1
    FILL_VALUE = 0.0

    slots = [None, None]

    if len(detection_result.hand_landmarks) == 0:
        return None

    for i in range(len(detection_result.hand_landmarks)):
        landmarks = detection_result.hand_landmarks[i]
        handedness = detection_result.handedness[i][0].category_name

        # Decide preferred slot
        preferred_slot = LEFT_SLOT if handedness.lower() == "left" else RIGHT_SLOT

        if slots[preferred_slot] is None:
            slots[preferred_slot] = landmarks
        else:
            other_slot = RIGHT_SLOT if preferred_slot == LEFT_SLOT else LEFT_SLOT
            if slots[other_slot] is None:
                slots[other_slot] = landmarks

    # Build CSV row
    row = []
    for landmarks in slots:
        if landmarks is not None:
            for landmark in landmarks: # type: ignore
                row.extend([landmark.x, landmark.y, landmark.z])
        else:
            # Fill missing hand
            row.extend([FILL_VALUE] * (21 * 3))

    return row

N_FRAMES = 25

if __name__ == "__main__":
    files = os.scandir("input")
    time = 0 # continuously running index to satisfy mediapipes need for a timestamp
    deq = deque(maxlen=N_FRAMES) # holds up to 25 frame_rows which correspond to the numbers in an actual csv row

    model = MP_model("hand_landmarker.task")
    model.init_video()

    csv_file = open("table.csv", "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_header = [f"{t}:{hand}_{axis}_{i}" for t in range(1-N_FRAMES, 1) for hand in ("left", "right") for i in range(21) for axis in ("x", "y", "z")]
    csv_header.append("sign")
    csv_writer.writerow(csv_header)

    for file in files:
        print("Processing " + file.name + "...")

        # SOMEHOW FIND THE LABEL FOR THIS VIDEO
        label = "spam"

        # Loop through each frame in the video
        cap = cv2.VideoCapture(file.path)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            mp_image = convert_frame_to_mp_image(frame)
            detection_result = model.landmarker.detect_for_video(mp_image, time)
            # draw result on the original frame (consider using mp_image.numpy_view() for viewing the image mediapipe actually works with)
            annotated_image = draw_landmarks_on_image(frame, detection_result)
            frame_row = build_frame_row(detection_result)

            if frame_row is not None:
                deq.append(frame_row)

            if len(deq) == N_FRAMES:
                combined_frame_rows = [entry for frame_rows in deq for entry in frame_rows]
                print(len(combined_frame_rows))
                combined_frame_rows.append(label)
                csv_writer.writerow(combined_frame_rows)

            cv2.imshow("Verification", annotated_image)
            cv2.waitKey(1) # opens the window and displays it for the given number of miliseconds
            
            time += 1

        # clear the deq to start collecting frames from the beginning again for new signs
        # ATTENTION: this means that signs that are shorter than N_FRAMES dont get recognized at all
        deq.clear()

        cap.release()
        cv2.destroyAllWindows()
        
    csv_file.close()