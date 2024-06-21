import cv2
import os

def frame_extractor(video_path: str, frame_path: str):
    video_capture = cv2.VideoCapture(video_path)
    if video_capture.isOpened():
        try:
            if not os.path.exists(frame_path):
                os.makedirs(frame_path)
        except OSError:
            print(f'Error creating file {frame_path}')

        frame_number = 0
        while(True):
            ret, frame = video_capture.read()
            if not ret:
                break
            save_filename = frame_path + '/' + str(frame_number) + '.jpg'
            cv2.imwrite(save_filename, frame)
            frame_number +=1
        video_capture.release()
        cv2.destroyAllWindows()
    else:
        print('Error opening video')

video_path = 'test_vid.mp4'
frame_path = 'videoframes'

frame_extractor(video_path=video_path, frame_path=frame_path)
