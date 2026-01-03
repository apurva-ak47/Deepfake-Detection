import cv2
import os

IMG_SIZE = (224, 224)
SAMPLE_EVERY_N = 5  # take every 5th frame
MAX_FRAMES = 100    # limit frames per video (optional)

def extract_and_save(video_path, out_folder, label):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open: {video_path}")
        return 0

    os.makedirs(out_folder, exist_ok=True)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved = 0
    frame_id = 0

    print(f"🎞️ Processing {label} video: {os.path.basename(video_path)} ({total} frames)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % SAMPLE_EVERY_N == 0:
            frame = cv2.resize(frame, IMG_SIZE)
            out_path = os.path.join(out_folder, f"{label}_{os.path.basename(video_path).split('.')[0]}_{saved:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        frame_id += 1
        if saved >= MAX_FRAMES:
            break

    cap.release()
    print(f"✅ Saved {saved} frames from {os.path.basename(video_path)}\n")
    return saved

if __name__ == "__main__":
    base_dir = "dataset_videos"
    output_dir = "dataset/train"

    for label in ["real", "fake"]:
        folder = os.path.join(base_dir, label)
        videos = [f for f in os.listdir(folder) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        for video in videos:
            video_path = os.path.join(folder, video)
            extract_and_save(video_path, os.path.join(output_dir, label), label)
