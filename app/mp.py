import mediapipe as mp

import cv2
import landmark_visualization

class MP_model:

    def __init__(self, path_to_model):
        self.model_path = path_to_model
        
    def init_livestream(self):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a hand landmarker instance with the live stream mode:
        def print_result(result, output_image: mp.Image, timestamp_ms: int):
            print('hand landmarker result: {}'.format(result))

            annotated_image = landmark_visualization.draw_landmarks_on_image(output_image.numpy_view(), result)
            cv2.imshow("imaeg", annotated_image)
            cv2.waitKey(1)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=print_result)
        
        self.landmarker = HandLandmarker.create_from_options(options)
        
    def init_video(self):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a hand landmarker instance with the video mode:
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2)
        
        self.landmarker = HandLandmarker.create_from_options(options)

    def init_image(self, image_path):
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a hand landmarker instance with the image mode:
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