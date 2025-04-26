import os
import time
import cv2
import ffmpeg
import argparse
from pathlib import Path
import shutil
from glob import glob
import numpy as np
import onnxruntime as ort
import threading

# ==== Configuration ====
pic_form = [".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"]
# input_video = "./inputs/video.mp4"
# checkpoint = "./checkpoint/paprika"
extract_dir = "extracts"
frame_dir = f"{extract_dir}\\frames"
style_dir = f"{extract_dir}\\styles"
audio_file = f"{extract_dir}\\audio.mp3"
# output_video = "anime_video.mp4"
# fps = 30  # Output video frame rate


# ==== Step 1: Extract frames from video and audio ====
def get_fps_fast(video_path):
    # using only opencv-python package, fast but can be inaccurate
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def extract_frames(input_video):
    (
        ffmpeg
        .input(input_video)
        .output(f"{frame_dir}/frame_%04d.jpg", qscale=2)
        .run(overwrite_output=True)
    )


def extract_audio(input_video):
    ffmpeg.input(input_video).output(audio_file, vn=None).run(overwrite_output=True)

# ==== Step 2: Change style of each photo ====
# def stylize_frames():
#     os.makedirs(style_dir, exist_ok=True)
#     frames = sorted(glob(f"{frame_dir}/*.jpg"))
#     print("🎨  Converting photo style...")
#     for frame_path in tqdm(frames):
#         filename = os.path.basename(frame_path)
#         output_path = os.path.join(style_dir, filename)
#         subprocess.run([
#             "python", "inference.py",
#             "--input", frame_path,
#             "--output", output_path,
#             "--checkpoint", checkpoint
#         ], stdout=subprocess.DEVNULL)


# def stylize_frames_all():
#     os.makedirs(style_dir, exist_ok=True)
#     frames = sorted(glob(f"{frame_dir}/*.jpg"))
#     print(f"{[len(frames)]}🎨  Converting photo style...")
#     process = subprocess.Popen([
#         # "python", "inference.py",
#         "python", "deploy/test_by_onnx.py",
#         "-i", frame_dir,
#         "-o", style_dir,
#         "-m", "deploy/animeganv3_H40_model.onnx"
#         # "--checkpoint", checkpoint
#     ], stdout=subprocess.PIPE, text=True)
#     for line in process.stdout:
#         print(line.strip())
#     process.wait()
#     print("🎨  Photo style conversion complete!")

# AnimeGANv3 Video Style Transfer
def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# select style model


def get_style_model():
    resources = "resources"
    onnx = "onnx"
    styles = ["AnimeGANv3_Hayao_36", "animeganv3_H40_model", "animeganv3_H64_model0", "AnimeGANv3_JP_face_v1.0",
              "AnimeGANv3_PortraitSketch_25", "AnimeGANv3_Shinkai_37", "AnimeGANv3_Shinkai_40", "AnimeGANv3_tiny_Cute"]
    # select style model
    print(f"Available styles to set: (default: {styles[0]})")
    for idx, key in enumerate(styles, start=1):
        print(f"{idx}. {key}")

    selections = input("Enter the numbers of the styles you want to set (e.g. 1 8): ")
    indexes = [int(i) for i in selections.split() if i.isdigit()]

    if (len(selections) > 0 and int(selections[0]) > 0 and int(selections[0]) <= len(styles)):
        for i in indexes:
            if 1 <= i <= len(styles):
                key = styles[i - 1]
                value = f"{resources}\{key}.onnx"
                # print(f"Selected style model: {value}")
                return value
    else:
        print(f"Selected style model default: {styles[0]}")
        return f"{resources}\{styles[0]}.onnx"


def process_image(img, model_name):
    h, w = img.shape[:2]
    # resize image to multiple of 8s

    def to_8s(x):
        # If using the tiny model, the multiple should be 16 instead of 8.
        if "tiny" in os.path.basename(model_name):
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


def convert_images(input_imgs_path, output_path, onnx="model.onnx", device="cpu"):
    result_dir = output_path
    check_folder(result_dir)
    test_files = glob("{}/*.*".format(input_imgs_path))
    test_files = [x for x in test_files if os.path.splitext(x)[-1] in pic_form]
    if ort.get_device() == "GPU" and device == "gpu":
        session = ort.InferenceSession(onnx, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    else:
        session = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
    x = session.get_inputs()[0].name
    y = session.get_outputs()[0].name

    begin = time.time()
    for i, sample_file in enumerate(test_files):
        t = time.time()
        sample_image, shape = load_test_data(sample_file, onnx)
        image_path = os.path.join(result_dir, "{0}".format(os.path.basename(sample_file)))
        fake_img = session.run(None, {x: sample_image})
        save_images(fake_img[0], image_path, (shape[1], shape[0]))
        print(f"Processing image: {i}, image size: {shape[1], shape[0]}, " + sample_file, f" time: {time.time() - t:.3f} s")
    end = time.time()
    print(f"Average time per image : {(end-begin)/len(test_files)} s")

# ==== Step 3: Recreate video from converted photos ====


def create_video(fps, output_video="out_style.mp4"):
    print("🎬  Reassembling anime video from frames...")
    # (
    #     ffmpeg
    #     .input(f"{style_dir}/frame_%04d.jpg", framerate=fps)
    #     .output(output_video, audio_file=audio_file, vcodec="libx264", acodec='aac', pix_fmt="yuv420p", shortest=None)
    #     .run(overwrite_output=True)
    # )
    image = ffmpeg.input(f"{style_dir}/frame_%04d.jpg", framerate=fps).video
    audio = ffmpeg.input(audio_file).audio
    ffmpeg_command = ffmpeg.output(image, audio, output_video, vcodec="libx264", acodec='mp3', pix_fmt="yuv420p", shortest=None)
    ffmpeg_command.run(overwrite_output=True)
    print(f"✅ Anime video created: {output_video}")

# ==== Step 4: Main function to run the pipeline ====


def main():
    parser = argparse.ArgumentParser(description="AnimeGANv3 Video Style Transfer")

    # Define parameters
    parser.add_argument("-i", "--input", help="Input video file")
    parser.add_argument("-fd", "--frames_dir", default="frames", help="Extract frames from video directory")
    parser.add_argument("-sd", "--style_dir", default="frames_out", help="Stylize frames directory")
    # parser.add_argument("-s", "--style",default="AnimeGANv3_Hayao_36.onnx", help="Style model path")
    parser.add_argument("-d", "--device", default="cpu", help="Device to use (cpu or gpu cuda)")
    parser.add_argument("-f", "--fps", type=int, help="fps of output video")
    parser.add_argument("-o", "--output", help="Output video file")

    # Parse the arguments
    args = parser.parse_args()

    # check if input video is None
    if (args.input is None):
        print("Error: Input video URL is required.")
        return
    #
    input_path = Path(args.input).parent.resolve()
    input_name = Path(args.input).stem
    input_ext = Path(args.input).suffix
    #
    shutil.rmtree(extract_dir, ignore_errors=True)
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(style_dir, exist_ok=True)

    print(f"Input video path: {args.input}")
    # check fps is None
    if (args.fps is None):
        args.fps = get_fps_fast(args.input)

    # check output video is None
    if (args.output is None):
        args.output = f"{input_path}\{input_name}_style{input_ext}"

    # Access parameters
    print(f"Url: {args.input}")
    print(f"frames_dir: {args.frames_dir}")
    print(f"style_dir: {args.style_dir}")
    # print(f"style: {resources}/{args.style}")
    print(f"device: {args.device}")
    print(f"fps: {args.fps}")
    print(f"output: {args.output}")

    # Extract frames from video
    print("📽️ Extracting frames from video...")
    extract_frames(args.input)
    print("📽️ Extracting frames from done")

    print("🎶 Extracting audio from video...")
    extract_audio(args.input)
    print("🎶 Extracting audio from done")
    # stylize_frames_all()

    # select style model
    selected_style = get_style_model()
    print(f"Selected style model: {selected_style}")

    # Create and start the thread
    thread_conver = threading.Thread(target=convert_images, args=(frame_dir, style_dir, selected_style, args.device))
    thread_conver.start()
    # Main thread continues
    print("Convert frame in background thread...")
    # Wait for thread to complete
    thread_conver.join()
    # Continue after thread is done
    print("Background convert thread done. next step...")
    # create video
    create_video(args.fps, args.output)


# ==== Run the entire pipeline ====
if __name__ == "__main__":
    main()

# ==== End of script ====