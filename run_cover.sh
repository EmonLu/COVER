#!/bin/bash

cd /home/jiehui/VQA/COVER
conda activate vqa-py312

video_folder=/home/jiehui/VQA/videos
LOG_FILE="evaluation_log.txt"

# 清空日志文件
> "$LOG_FILE"

# 遍历 videos 文件下的所有 mp4 文件
for video_file in "$video_folder"/*.mp4; do
    if [ -f "$file" ]; then
        echo "Processing file: $video_file"
    fi
    CMD="python evaluate_one_video.py -v $video_file"
    
    echo "Running command: $CMD" >> "$LOG_FILE"
    
    $CMD 2>&1 | grep "seconds" >> "$LOG_FILE"
    
     
    echo "----------------------------------------" >> "$LOG_FILE"
done

for video_file in "$video_folder"/*.mp4; do
    if [ -f "$file" ]; then
        echo "Processing file: $video_file"
    fi
    CMD="numactl -C 0-15 python evaluate_one_video.py -v $video_file"
    
    echo "Running command: $CMD" >> "$LOG_FILE"
    
    $CMD 2>&1 | grep "seconds" >> "$LOG_FILE"
    
    echo "----------------------------------------" >> "$LOG_FILE"
done

for video_file in "$video_folder"/*.mp4; do
    if [ -f "$file" ]; then
        echo "Processing file: $video_file"
    fi
    CMD="numactl -C 0-23 python evaluate_one_video.py -v $video_file"
    
    echo "Running command: $CMD" >> "$LOG_FILE"
    
    $CMD 2>&1 | grep "seconds" >> "$LOG_FILE"
    
     
    echo "----------------------------------------" >> "$LOG_FILE"
done

echo "Evaluation completed. Check $LOG_FILE for details."