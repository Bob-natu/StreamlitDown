import streamlit as st
import cv2
import mediapipe as mp
import tempfile
import io

# MediaPipeのPoseモジュールを初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

st.title("動画の骨格抽出と再生")

# 動画アップロード
uploaded_file = st.file_uploader("動画ファイルをアップロードしてください", type=["mp4", "mov", "avi"])
if uploaded_file is not None:
    # アップロードされたファイルを一時ディレクトリに保存
    temp_input_file = tempfile.NamedTemporaryFile(delete=False)
    temp_input_file.write(uploaded_file.read())
    
    # 入力動画のパス
    input_video_path = temp_input_file.name
    
    # 処理済み動画を保存する一時ファイル
    temp_output_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    output_video_path = temp_output_file.name

    # 動画を読み込む
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 動画の書き込み設定
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4形式
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # MediaPipe Poseを使ってフレームごとに骨格を抽出
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # フレームをRGBに変換
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 骨格を検出
            results = pose.process(image)

            # 骨格を描画
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                )

            # 書き込み用のフレームをBGRに戻して保存
            out.write(frame)

    # 処理終了後にリソースを解放
    cap.release()
    out.release()

    # 処理済み動画をBytesIOオブジェクトに読み込む
    with open(output_video_path, "rb") as f:
        video_bytes = f.read()

    # Streamlitで処理済み動画を再生
    st.video(io.BytesIO(video_bytes))

    # 一時ファイルを削除（オプション）
    temp_input_file.close()
    temp_output_file.close()
