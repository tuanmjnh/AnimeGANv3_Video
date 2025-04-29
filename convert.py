import os
import time
import os
import cv2
from glob import glob
import threading
import subprocess
import numpy as np
import onnxruntime as ort
import tkinter as tk
from tkinter import filedialog

# ==== Cấu hình ====
pic_form = ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG']
input_video = "./inputs/video.mp4"
checkpoint = "./checkpoint/paprika"
extract_dir = "extracts"
frame_dir = f"{extract_dir}\\frames"
style_dir = f"{extract_dir}\\styles"
audio_file = f"{extract_dir}\\audio.mp3"
output_video = "anime_video.mp4"
fps = 30  # Tốc độ khung hình của video đầu ra

# upload video file
def upload_file(file_type):
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        title=f"Select {file_type} file",
        filetypes=[(f"{file_type.upper()} files", f"*.{file_type.lower()}"), ("All files", "*.*")]
    )
    return file_path


def stylize_frames_all():
    os.makedirs(style_dir, exist_ok=True)
    frames = sorted(glob(f"{frame_dir}/*.jpg"))
    print(f"{[len(frames)]}🎨  Đang chuyển đổi phong cách ảnh...")
    process = subprocess.Popen([
        # "python", "inference.py",
        "python", "resources/test_by_onnx.py",
        "-i", frame_dir,
        "-o", style_dir,
        "-m", "resources/animeganv3_H40_model.onnx"
        # "--checkpoint", checkpoint
    ], stdout=subprocess.PIPE, text=True)
    for line in process.stdout:
        print(line.strip())
    process.wait()
    print("🎨  Đã chuyển đổi phong cách ảnh xong!")


def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def process_image(img, model_name):
    h, w = img.shape[:2]
    # resize image to multiple of 8s

    def to_8s(x):
        # If using the tiny model, the multiple should be 16 instead of 8.
        if 'tiny' in os.path.basename(model_name):
            return 256 if x < 256 else x - x % 16
        else:
            return 256 if x < 256 else x - x % 8
    img = cv2.resize(img, (to_8s(w), to_8s(h)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
    return img


def load_test_data(image_path, model_name):
    img0 = cv2.imread(image_path).astype(np.float32)
    img = process_image(img0, model_name)
    img = np.expand_dims(img, axis=0)
    return img, img0.shape


def save_images(images, image_path, size):
    images = (np.squeeze(images) + 1.) / 2 * 255
    images = np.clip(images, 0, 255).astype(np.uint8)
    images = cv2.resize(images, size)
    cv2.imwrite(image_path, cv2.cvtColor(images, cv2.COLOR_RGB2BGR))


def Convert(input_imgs_path, output_path, onnx="model.onnx", device="cpu"):
    # result_dir = opj(output_path, style_name)
    result_dir = output_path
    check_folder(result_dir)
    test_files = glob('{}/*.*'.format(input_imgs_path))
    test_files = [x for x in test_files if os.path.splitext(x)[-1] in pic_form]
    print(ort.get_available_providers())
    if ort.get_device() == 'GPU' and device == "gpu":
        session = ort.InferenceSession(onnx, providers=['TensorrtExecutionProvider', 'CPUExecutionProvider',])  # CUDAExecutionProvider
    else:
        session = ort.InferenceSession(onnx, providers=['CPUExecutionProvider', ])
    x = session.get_inputs()[0].name
    y = session.get_outputs()[0].name

    begin = time.time()
    for i, sample_file in enumerate(test_files):
        t = time.time()
        sample_image, shape = load_test_data(sample_file, onnx)
        image_path = os.path.join(result_dir, '{0}'.format(os.path.basename(sample_file)))
        fake_img = session.run(None, {x: sample_image})
        save_images(fake_img[0], image_path, (shape[1], shape[0]))
        print(f'Processing image: {i}, image size: {shape[1], shape[0]}, ' + sample_file, f' time: {time.time() - t:.3f} s')
    end = time.time()
    print(f'Average time per image : {(end-begin)/len(test_files)} s')


def run_inference():
    # Chạy hàm Convert với các tham số đã định nghĩa
    Convert(frame_dir, style_dir, "resources/AnimeGANv3_Hayao_36.onnx", "gpu")


video_path = upload_file("mp4")
if (video_path is None or video_path == ""):
    print("No video file selected. Exiting...")
    exit(0)

# Create and start the thread
thread_conver = threading.Thread(target=run_inference)
thread_conver.start()

# Main thread continues
print("Convert frame in background thread...")

# Wait for thread to complete
thread_conver.join()

# Continue after thread is done
print("Background convert thread done. next step...")
