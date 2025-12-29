#import cv2
#import landmark_visualization

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
import numpy as np
import cv2

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class MP_model:
    def __init__(self, path_to_model: str):
        """
        Initializes the path to the MediaPipe model for later use in video/image initializations.
        
        :param path_to_model: Path to the .task-model.
        :type path_to_model: str
        """
        self.model_path = path_to_model
     
    def init_livestream(self):
        """Initializes MediaPipe with VisionRunningMode=LIVE_STREAM."""

        def print_result(result, output_image: mp.Image, timestamp_ms: int):
            print('hand landmarker result: {}'.format(result))
            #annotated_image = landmark_visualization.draw_landmarks_on_image(output_image.numpy_view(), result)
            #cv2.imshow("imaeg", annotated_image)
            #cv2.waitKey(1)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=print_result)
        
        self.landmarker = HandLandmarker.create_from_options(options)
        
    def init_video(self):
        """Initializes MediaPipe with VisionRunningMode=VIDEO."""

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2)
        
        self.landmarker = HandLandmarker.create_from_options(options)

    """
    def init_image(self, image_path):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2)
        self.landmarker = HandLandmarker.create_from_options(options)

        image = mp.Image.create_from_file(image_path)
        result = self.landmarker.detect(image)
        print(result)
        annotated_image = landmark_visualization.draw_landmarks_on_image(image.numpy_view(), result)
        cv2.imshow("test", annotated_image)
        cv2.waitKey()
    """

def draw_landmarks_on_image(rgb_image, detection_result):
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([ # type: ignore
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks # type: ignore
        ])
    
        solutions.drawing_utils.draw_landmarks( # type: ignore
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS, # type: ignore
            solutions.drawing_styles.get_default_hand_landmarks_style(), # type: ignore
            solutions.drawing_styles.get_default_hand_connections_style()) # type: ignore

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image
