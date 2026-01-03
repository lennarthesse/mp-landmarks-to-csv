import cv2
import os
import mp
import mediapipe
import landmark_visualization

if __name__ == "__main__":
    files = os.scandir("input")

    model = mp.MP_model("hand_landmarker.task")
    model.init_video()

    time = 0

    for file in files:
        print("Processing " + file.name + "...")

        cap = cv2.VideoCapture(file.path)

        # Loop through each frame in the video
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # Convert OpenCV frame from BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert the frame to a MediaPipe Image object
            mp_image = mediapipe.Image(
                image_format=mediapipe.ImageFormat.SRGB,
                data=frame_rgb
            )

            result = model.landmarker.detect_for_video(mp_image, time)
            print(result)

            annotated_image = landmark_visualization.draw_landmarks_on_image(mp_image.numpy_view(), result)

            time += 1

            cv2.imshow("Webcam", annotated_image)
            cv2.waitKey(1)
            
        cap.release()
        cv2.destroyAllWindows()