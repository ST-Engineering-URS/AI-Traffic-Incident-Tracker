import cv2
import os

# 输入视频路径
video_path = 'C:/Users/admin/Downloads/videoplayback (3).mp4'   
# 输出图片目录
output_dir = r'C:/Users/admin/Desktop/accident_photos/3'  

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 打开视频
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("无法打开视频")
    exit()

# 获取视频的帧率
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"视频帧率: {fps}")

# 每隔多少帧取一张
frame_interval = 10

frame_count = 0  # 视频当前帧数
save_count = 4000  # 保存图片起始编号

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = os.path.join(output_dir, f"{save_count}.jpg")
        cv2.imwrite(filename, frame)
        print(f"保存: {filename}")
        save_count += 1

    frame_count += 1

cap.release()
print("完成！")
