import cv2
import os

def extract_frames(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    count = 0
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_filename = os.path.join(output_dir, f'{count:05d}.jpg')
            cv2.imwrite(frame_filename, frame)
            count += 1
        else:
            break
    
    cap.release()

def process_videos(dataset_dir):

    raw_dir = os.path.join(dataset_dir, 'raw')
    
    if os.path.exists(raw_dir) and os.path.isdir(raw_dir):
        for video_file in os.listdir(raw_dir):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(raw_dir, video_file)
                output_dir_name = video_file.rsplit('.', 1)[0]  
                output_dir = os.path.join(dataset_dir, output_dir_name)
                extract_frames(video_path, output_dir)

if __name__ == "__main__":
    dataset_dir = 'dataset'
    process_videos(dataset_dir)
