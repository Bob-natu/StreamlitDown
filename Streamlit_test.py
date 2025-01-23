import cv2
import streamlit as st
import tempfile
import os

# 動画の読み込みと処理（例：骨格抽出の処理）
# 出力動画ファイルを作成
output_video_path = '/path/to/output_video.mp4'
cap = cv2.VideoCapture('input_video.mp4')
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 出力動画の準備
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ここで骨格抽出を行い、フレームに描画
    # (骨格抽出処理のコードは省略)

    # 動画にフレームを書き込み
    out.write(frame)

cap.release()
out.release()

# 動画を一時的なファイルとして保存
with open(output_video_path, 'rb') as f:
    video_bytes = f.read()

# Streamlitで動画を表示
st.video(video_bytes)
