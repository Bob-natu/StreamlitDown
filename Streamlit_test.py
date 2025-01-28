import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import tempfile
import matplotlib.pyplot as plt

# Streamlit アプリの設定
st.title("スパイク解析: 骨格と進行グラフ")
st.sidebar.header("設定")

# ファイルアップロード
uploaded_file = st.file_uploader("動画ファイルをアップロードしてください", type=["mp4","avi", "mov"])

if uploaded_file is not None:
    # 一時ディレクトリの作成と安全な管理
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # 動画ファイルを保存
            input_video_path = os.path.join(temp_dir, uploaded_file.name)
            with open(input_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.sidebar.success("動画がアップロードされました。解析を開始します。")

            # MediaPipe Poseの設定
            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils

            # データ保存用
            frame_numbers = []
            right_shoulder_y = []
            left_shoulder_y = []
            min_right_wrist_y = float('inf')
            highest_wrist_image = None
            highest_frame_number = None

            # 動画の読み込み
            cap = cv2.VideoCapture(input_video_path)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 出力動画設定（動画の右側にグラフを追加するために幅を増加）
            output_width = frame_width + 400  # フレームに余白を追加
            output_height = frame_height
            output_video_path = os.path.join(temp_dir, "output_video_with_skeleton_and_graph.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # エンコーディングフォーマット
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height))

            # 進捗バー
            progress_bar = st.progress(0)

            # Poseインスタンスの作成
            with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # フレームをRGBに変換してMediaPipeで処理
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)

                    # フレーム番号の取得
                    frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks.landmark
                        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

                        # データの記録
                        frame_numbers.append(frame_number)
                        right_shoulder_y.append(right_shoulder.y)
                        left_shoulder_y.append(left_shoulder.y)

                        # 手首の最高点の記録
                        if right_wrist.y < min_right_wrist_y:
                            min_right_wrist_y = right_wrist.y
                            highest_wrist_image = frame.copy()
                            highest_frame_number = frame_number

                        # 骨格の描画
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    # グラフの作成（進行中）
                    fig, ax = plt.subplots(figsize=(4, 3))  # グラフのサイズを調整
                    ax.plot(frame_numbers, [1 - y for y in right_shoulder_y], label="Right Shoulder Y", color="blue")
                    ax.plot(frame_numbers, [1 - y for y in left_shoulder_y], label="Left Shoulder Y", color="green")
                    ax.set_xlim(0, total_frames)
                    ax.set_xlabel("Frame Number")
                    ax.set_ylabel("Normalized Y Coordinate")
                    ax.legend()
                    plt.tight_layout()

                    # グラフを画像として保存
                    graph_image_path = os.path.join(temp_dir, "graph_image.png")
                    fig.savefig(graph_image_path, format="png", dpi=100)
                    plt.close(fig)

                    # グラフ画像を読み込み
                    graph_image = cv2.imread(graph_image_path)
                    graph_resized = cv2.resize(graph_image, (400, frame_height))  # グラフを縦いっぱいにリサイズ

                    # フレームにグラフを合成
                    combined_frame = np.zeros((frame_height, output_width, 3), dtype=np.uint8)
                    combined_frame[:, :frame_width] = frame  # 左側に元のフレーム
                    combined_frame[:, frame_width:] = graph_resized  # 右側にグラフ

                    # フレームを保存
                    out.write(combined_frame)

                    # 進捗バーの更新
                    progress = int((frame_number / total_frames) * 100)
                    progress_bar.progress(progress)

            # リソース解放
            cap.release()
            out.release()

            # 動画解析完了
            st.success("解析が完了しました！")
            progress_bar.empty()

            # ダウンロードリンクを提供
            with open(output_video_path, "rb") as f:
                st.download_button(
                    label="解析した動画をダウンロード",
                    data=f,
                    file_name="output_video_with_skeleton_and_graph.mp4",
                    mime="video/mp4"
                )

            # 完成したグラフを表示
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(frame_numbers, [1 - y for y in right_shoulder_y], label="Right Shoulder Y", color="blue")
            ax.plot(frame_numbers, [1 - y for y in left_shoulder_y], label="Left Shoulder Y", color="green")
            ax.set_xlabel("Frame Number")
            ax.set_ylabel("Normalized Y Coordinate")
            ax.set_title("Shoulder and Wrist Positions Over Time")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

            # 右手首の最高到達点のフレームを表示
            if highest_wrist_image is not None:
                st.image(
                    highest_wrist_image, 
                    caption=f"右手首が最高到達点に達したフレーム: {highest_frame_number}",
                    use_container_width=True,  # Updated parameter
                    channels="BGR"
                )

        except Exception as e:
            st.error(f"エラーが発生しました: {e}")
