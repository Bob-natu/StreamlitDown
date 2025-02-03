import cv2 as cv
import mediapipe as mp
import tempfile

# MediaPipe Poseのセットアップ
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 一時ファイルに保存（ローカル確認用）
output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
print(f"Output path for verification: {output_path}")

# 動画の読み込み
video = cv.VideoCapture("入力ファイルパス.mp4")  # テストしたい動画ファイルを指定

frame_width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv.CAP_PROP_FPS)

# 出力動画の初期化
out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    out.write(frame)

video.release()
out.release()
pose.close()
