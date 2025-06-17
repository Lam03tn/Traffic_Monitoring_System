#!/bin/bash

# Số lần restart tối đa
MAX_RESTARTS=5
RESTART_COUNT=0

# Lệnh ffmpeg để giả lập camera
FFMPEG_CMD="ffmpeg -re -stream_loop -1 -i Camera_Simulator_Stream/videos/cam1.mp4 -c:v libx264 -preset veryfast -f rtsp rtsp://localhost:554/stream/cam1"

echo "🚀 Bắt đầu chạy FFmpeg RTMP stream (cam1)..."

while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    echo "👉 Lần chạy thứ $((RESTART_COUNT + 1))..."
    $FFMPEG_CMD

    EXIT_CODE=$?
    echo "⚠️ FFmpeg exited với mã $EXIT_CODE"

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ FFmpeg kết thúc bình thường. Dừng script."
        break
    fi

    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo "🔁 Khởi động lại sau 3 giây..."
    sleep 3
done

if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
    echo "❌ Đã đạt đến số lần restart tối đa ($MAX_RESTARTS). Dừng script."
fi
