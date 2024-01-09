import os
import cv2

def save_sequence(sequence_path: str, video_path: str):
    images = sorted([img for img in os.listdir(sequence_path) if img.endswith(".jpg")])
    frame = cv2.imread(os.path.join(sequence_path, images[0]))
    height, width, _ = frame.shape

    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 60, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(sequence_path, image)))

    cv2.destroyAllWindows()
    video.release()


# VisDrone
SOURCE_BASE = "/home/reem/Desktop/UAV & satellite datasets/VisDrone2019-VID-train/sequences/"
# SAVE_BASE = "/home/reem/Desktop/UAV & satellite datasets/videos_waldo"
SAVE_BASE = "/home/reem/Desktop/WALDO/playground/input_vids"
for folder in os.listdir(SOURCE_BASE):
    save_sequence(SOURCE_BASE + folder, f'{SAVE_BASE}/{folder}.mp4')

