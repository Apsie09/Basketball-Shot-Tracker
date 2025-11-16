#!/bin/bash
# Download basketball video using yt-dlp

if [ $# -eq 0 ]; then
    echo "Usage: $0 <youtube_url>"
    exit 1
fi

URL=$1
OUTPUT_DIR="../data/videos"

mkdir -p $OUTPUT_DIR

yt-dlp -f "bestvideo[height=720]+bestaudio" \
       --merge-output-format mp4 \
       -o "$OUTPUT_DIR/basketball_%(title)s.%(ext)s" \
       "$URL"

echo "Video downloaded to $OUTPUT_DIR"