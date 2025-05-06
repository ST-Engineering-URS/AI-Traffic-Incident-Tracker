import tkinter as tk
from PIL import Image, ImageTk
import os

# 参数设置
folders = [
    r"C:/Users/admin/Desktop/accident_photos/3",
    r"C:/Users/admin/Desktop/accident_photos/4",
    r"C:/Users/admin/Desktop/accident_photos/5",
    r"C:/Users/admin/Desktop/comparison_photos/2",
    r"C:/Users/admin/Desktop/comparison_photos/6",
    r"C:/Users/admin/Desktop/comparison_photos/7",
]
rows, cols = 2, 3  # 2行3列布局
fps = 6
interval = int(1000 / fps)  # 每帧间隔（毫秒）
thumb_size = (200, 250)  # 缩略图大小，可根据窗口调整

# 更新帧的函数
def update_frame(label, frames, idx=0):
    frame = frames[idx]
    label.config(image=frame)
    label.image = frame
    label.after(interval, update_frame, label, frames, (idx + 1) % len(frames))

# 全屏查看函数
def open_fullscreen(image_path, root):
    img = Image.open(image_path)
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    img = img.resize((screen_w, screen_h), Image.Resampling.LANCZOS)
    tk_img = ImageTk.PhotoImage(img)

    win = tk.Toplevel(root)
    win.attributes("-fullscreen", True)
    win.configure(background='black')

    canvas = tk.Canvas(win, width=screen_w, height=screen_h, highlightthickness=0)
    canvas.pack()
    canvas.create_image(0, 0, anchor='nw', image=tk_img)
    canvas.image = tk_img
    status_text = "救护车：无  消防车：无  事故：发生"
    canvas.create_text(
        screen_w - 20, 20,
        text=status_text,
        anchor='ne',
        fill='white',
        font=('Arial', 24, 'bold')
    )
    win.bind("<Escape>", lambda e: win.destroy())

if __name__ == '__main__':
    root = tk.Tk()
    root.title("3x2 相框播放")
    # 可根据显示器设置大小
    root.geometry(f"{cols * (thumb_size[0] + 10)}x{rows * (thumb_size[1] + 10)}")

    # 创建标签并启动各自的帧动画
    for idx, folder in enumerate(folders):
        # 获取并排序图片文件
        image_files = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))
        ])
        frames = []
        for path in image_files:
            img = Image.open(path).resize(thumb_size, Image.Resampling.LANCZOS)
            frames.append(ImageTk.PhotoImage(img))

        # 计算位置
        r = idx // cols
        c = idx % cols
        lbl = tk.Label(root)
        lbl.grid(row=r, column=c, padx=5, pady=5)

        # 绑定点击事件，全屏显示第一张
        if image_files:
            lbl.bind("<Button-1>", lambda e, p=image_files[0]: open_fullscreen(p, root))

        # 启动动画循环
        if frames:
            update_frame(lbl, frames)

    root.mainloop()
