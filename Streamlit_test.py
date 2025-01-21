import streamlit as st
import cv2
import os
import tempfile
import mediapipe as mp
import io

# 動画アップロード
uploaded_file = st.file_uploader("動画ファイルをアップロードしてください", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_input_file:
        temp_input_file.write(uploaded_file.read())
        temp_input_file.flush()

        st.sidebar.success("動画がアップロードされました。解析を開始します。")

        # MediaPipe Poseの設定
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        # 動画の読み込み
        cap = cv2.VideoCapture(temp_input_file.name)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # メモリ上に出力動画を保存
        output_video = io.BytesIO()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4vを使用
        out = cv2.VideoWriter(temp_input_file.name, fourcc, fps, (frame_width, frame_height))

        # 進捗バー
        progress_bar = st.progress(0)

        # Poseインスタンスの作成
        with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
            frame_counter = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 骨格抽出処理
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    # 骨格を描画
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                else:
                    st.warning(f"フレーム {int(cap.get(cv2.CAP_PROP_POS_FRAMES))} で骨格が検出されませんでした")

                # 処理したフレームを書き込む
                out.write(frame)

                # 進捗バーの更新
                frame_counter += 1
                progress = int((frame_counter / total_frames) * 100)
                progress_bar.progress(progress)

            # リソース解放
            cap.release()
            out.release()

        # メモリに保存された動画を読み込む
        with open(temp_input_file.name, "rb") as video_file:
            video_bytes = video_file.read()

        output_video.seek(0)  # バッファを先頭に移動

        # 出力動画の表示
        st.success("動画の処理が完了しました！")
        st.video(io.BytesIO(video_bytes))  # メモリから動画を読み込み、表示

        # 一時ファイルのクリーンアップ
        os.remove(temp_input_file.name)
