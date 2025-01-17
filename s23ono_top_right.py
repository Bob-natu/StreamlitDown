# すでにインポートされているモジュール
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import os

# 入出力パス設定
input_video_path = r'C:\Users\bob\Documents\PythonScripts\movies\spike_15.mp4'
output_video_path = r'C:\Users\bob\Documents\PythonScripts\output\spike_15\output_spike_15.mp4'
output_graph_path = r'C:\Users\bob\Documents\PythonScripts\output\spike_15\graph_spike_15.jpg'
output_wrist_image_path = r'C:\Users\bob\Documents\PythonScripts\output\spike_15\spike_15_highest_wrist.jpg'

# 出力ディレクトリの作成
output_dir = os.path.dirname(output_video_path)
os.makedirs(output_dir, exist_ok=True)

# MediaPipe Pose 初期化
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# データ保存用
frame_numbers = []
right_shoulder_y = []
left_shoulder_y = []

# 右手首と右肩の最高到達点データ
highest_wrist_image = None
min_right_wrist_y = float('inf')
min_right_wrist_frame = -1

min_right_shoulder_y = float('inf')  # 右肩の最高到達点（最小y座標）
highest_shoulder_frame = -1  # 最高到達点のフレーム番号

# 動画読み込み
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise FileNotFoundError(f"動画ファイルを開けません: {input_video_path}")

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 出力動画設定
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
if not out.isOpened():
    raise FileNotFoundError(f"出力動画を作成できません: {output_video_path}")

# グラフの初期設定
plt.ion()  # インタラクティブモード有効化
fig, ax = plt.subplots(figsize=(10, 5))
line_right, = ax.plot([], [], label="Right Shoulder Y", color="blue")
line_left, = ax.plot([], [], label="Left Shoulder Y", color="green")
highest_point, = ax.plot([], [], 'ro', label="Highest Right Shoulder")
ax.set_xlim(0, total_frames)
ax.set_ylim(0, 1)
ax.set_xlabel("Frame Number")
ax.set_ylabel("Y Coordinate (Flipped)")
ax.set_title("Shoulder Coordinates Over Time")
ax.legend()

# Pose インスタンス作成
with mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("動画処理が完了しました。")
            break

        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # フレームをRGBに変換して処理
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

            # データ保存
            frame_numbers.append(frame_number)
            right_shoulder_y.append(right_shoulder.y)
            left_shoulder_y.append(left_shoulder.y)

            # 右肩の最高到達点を記録
            if right_shoulder.y < min_right_shoulder_y:
                min_right_shoulder_y = right_shoulder.y
                highest_shoulder_frame = frame_number

            # 右手首の最高到達点を記録
            if right_wrist.y < min_right_wrist_y:
                min_right_wrist_y = right_wrist.y
                min_right_wrist_frame = frame_number
                highest_wrist_image = frame.copy()

            # 骨格を描画（ランドマークを使って描画）
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 肩に点を描画
            for (x, y, color) in [
                (right_shoulder.x, right_shoulder.y, (255, 0, 0)),  # 青
                (left_shoulder.x, left_shoulder.y, (0, 255, 0))   # 緑
            ]:
                cv2.circle(frame, (int(x * frame_width), int(y * frame_height)), 5, color, -1)

        # グラフを更新
        line_right.set_data(frame_numbers, [1 - y for y in right_shoulder_y])
        line_left.set_data(frame_numbers, [1 - y for y in left_shoulder_y])
        highest_point.set_data([highest_shoulder_frame], [1 - min_right_shoulder_y])  # 最高到達点を点で表示
        ax.set_xlim(0, max(10, frame_number + 10))  # フレーム数に応じて動的に更新
        plt.pause(0.01)  # 描画の更新

        # 進捗バー描画
        progress = int((frame_number / total_frames) * frame_width)
        cv2.rectangle(frame, (0, frame_height - 20), (progress, frame_height), (0, 255, 0), -1)

        # フレーム番号表示
        cv2.putText(frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 動画保存
        out.write(frame)

        # フレーム表示
        cv2.imshow("Processed Video", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            print("処理が中断されました。")
            break

    # リソース解放
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# グラフ保存
plt.ioff()  # インタラクティブモード終了
if frame_numbers:
    plt.savefig(output_graph_path)
    print(f"グラフ画像が保存されました: {output_graph_path}")
    plt.close()

# 右手首画像保存
if highest_wrist_image is not None:
    try:
        os.makedirs(os.path.dirname(output_wrist_image_path), exist_ok=True)
        success = cv2.imwrite(output_wrist_image_path, highest_wrist_image)
        if success:
            print(f"右手首の最高到達点画像が保存されました: {output_wrist_image_path}")
        else:
            print(f"画像保存に失敗しました: {output_wrist_image_path}")
    except Exception as e:
        print(f"右手首画像保存中にエラーが発生しました: {e}")
else:
    print("highest_wrist_image is None. No frame captured for the highest wrist.")
