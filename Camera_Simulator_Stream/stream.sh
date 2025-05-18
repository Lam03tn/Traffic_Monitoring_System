#!/bin/sh

MAX_RESTARTS=5
RESTART_COUNT=0
INPUT_FILE="/videos/$VIDEO_FILE"
STREAM_URL="rtmp://admin:admin@$HAPROXY_IP/stream/$STREAM_KEY"

echo "🚀 Stream từ $INPUT_FILE lên $STREAM_URL"

while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    echo "👉 Lần chạy thứ $((RESTART_COUNT + 1))..."
    ffmpeg -re -stream_loop -1 -i "$INPUT_FILE" -c:v libx264 -preset veryfast -b:v 2M -maxrate 2M -bufsize 4M -f flv "$STREAM_URL"
    
    EXIT_CODE=$?
    echo "⚠️ FFmpeg exited với mã $EXIT_CODE"

    [ $EXIT_CODE -eq 0 ] && break

    RESTART_COUNT=$((RESTART_COUNT + 1))
    echo "🔁 Khởi động lại sau 3 giây..."
    sleep 3
done

[ $RESTART_COUNT -ge $MAX_RESTARTS ] && echo "❌ Quá số lần restart. Dừng container."
