import streamlit as st
import cv2
import mediapipe as mp
import math
import matplotlib.pyplot as plt
import os

# Streamlit UI設定
st.title("肩の位置追跡とグラフ作成")

# 動画アップロード
uploaded_file = st.file_uploader("動画ファイルをアップロードしてください", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # 動画の読み込み
    input_video_path = os.path.join("/tmp", uploaded_file.name)
    with open(input_video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # 出力設定
    output_video_path = os.path.join("/tmp", "output.mp4")
    
    # MediaPipe Pose 初期化
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    
    # データ保存用
    frame_numbers = []
    right_shoulder_y = []
    left_shoulder_y = []
    right_wrist_y = []
    
    # 右肩の最高到達点データ
    min_right_shoulder_y = float('inf')  # 右肩の最小Y座標（最高到達点）
    min_right_shoulder_frame = -1  # 最高到達点のフレーム番号
    
    # 右手首の最高到達点データ
    min_right_wrist_y = float('inf')  # 右手首の最小Y座標（最高到達点）
    min_right_wrist_frame = -1  # 最高到達点のフレーム番号
    
    # 動画読み込み
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 新しい動画フレームの幅を設定（動画の右側にスペースを作成）
    new_frame_width = frame_width + (frame_width // 2)  # 右半分をグラフ用に空ける
    
    # 出力動画設定（新しい幅を使用）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (new_frame_width, frame_height))
    
    # グラフの初期設定
    plt.ion()
    fig, ax = plt.subplots(figsize=(5, 3))
    line_right, = ax.plot([], [], label="Right Shoulder Y", color="blue", alpha=0.7)
    line_left, = ax.plot([], [], label="Left Shoulder Y", color="green", alpha=0.7)
    highest_right_shoulder, = ax.plot([], [], 'ro', label="Highest Right Shoulder", alpha=0.7)
    highest_right_wrist, = ax.plot([], [], 'go', label="Highest Right Wrist", alpha=0.7)
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Y Coordinate (Flipped)")
    ax.legend()
    
    with mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
                
                frame_numbers.append(frame_number)
                right_shoulder_y.append(right_shoulder.y)
                left_shoulder_y.append(left_shoulder.y)
                right_wrist_y.append(right_wrist.y)
                
                # 右肩の最高到達点を記録
                if right_shoulder.y < min_right_shoulder_y:
                    min_right_shoulder_y = right_shoulder.y
                    min_right_shoulder_frame = frame_number
                
                # 右手首の最高到達点を記録
                if right_wrist.y < min_right_wrist_y:
                    min_right_wrist_y = right_wrist.y
                    min_right_wrist_frame = frame_number
                
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # グラフ更新
            line_right.set_data(frame_numbers, [1 - y for y in right_shoulder_y])
            line_left.set_data(frame_numbers, [1 - y for y in left_shoulder_y])
            if min_right_shoulder_frame > 0:
                highest_right_shoulder.set_data([min_right_shoulder_frame], [1 - min_right_shoulder_y])
            if min_right_wrist_frame > 0:
                highest_right_wrist.set_data([min_right_wrist_frame], [1 - min_right_wrist_y])
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.001)
            
            # グラフを画像化して新しいサイズのフレームに合成
            graph_image_path = "/tmp/graph_frame.jpg"
            plt.savefig(graph_image_path)
            graph_image = cv2.imread(graph_image_path)

            if graph_image is not None:
                # グラフ画像をリサイズ
                graph_resized = cv2.resize(graph_image, (frame_width // 2, frame_height // 2))
                
                # リサイズされたグラフの高さと幅を取得
                h, w, _ = graph_resized.shape
                
                # グラフをフレームの右側に配置 (右端に合わせて配置)
                if frame_width - w - 10 >= 0:
                    frame[0:h, frame_width - w - 10:frame_width - 10] = graph_resized
                else:
                    # フレームにグラフが収まらない場合の処理（例: エラーメッセージ）
                    print("グラフがフレームに収まりません。サイズを調整してください。")
            
            # 右側にグラフを貼り付けた新しい動画フレームを出力
            out.write(frame)
            
        cap.release()
        out.release()
        plt.ioff()
    
    # 動画ダウンロードボタン
    with open(output_video_path, "rb") as f:
        st.download_button("動画をダウンロード", f, file_name="processed_video.mp4", mime="video/mp4")
    
    # 右肩と右手首の最高到達点のフレームを表示
    st.write(f"右肩の最高到達点のフレーム: {min_right_shoulder_frame}")
    st.write(f"右手首の最高到達点のフレーム: {min_right_wrist_frame}")
