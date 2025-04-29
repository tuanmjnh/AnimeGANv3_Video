from os import makedirs
from cv2 import VideoCapture, CAP_PROP_FPS
from argparse import ArgumentParser
from shutil import rmtree
import libFFMPEG
import libAnimeGANv3
from pathlib import Path
from libIO import fileOpenDialog


# ==== Configuration ====
pic_form = [".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"]
# input_video = "./inputs/video.mp4"
# checkpoint = "./checkpoint/paprika"
absolute_path = Path().resolve()
resources_path = f"{absolute_path}\\Resources"
temporary_dir = f"{absolute_path}\\temporary"
frame_dir = f"{temporary_dir}\\frames"
style_dir = f"{temporary_dir}\\styles"
audio_file = f"{temporary_dir}\\audio.mp3"
# output_video = "anime_video.mp4"
# fps = 30  # Output video frame rate

# ==== Step 1: Extract frames from video and audio ====


def get_fps_fast(video_path):
    # using only opencv-python package, fast but can be inaccurate
    cap = VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open {video_path}")
    fps = cap.get(CAP_PROP_FPS)
    cap.release()
    return fps


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


# ==== Step 3: Recreate video from converted photos ====


# def create_video(fps, output_video="out_style.mp4"):
#     print("🎬  Reassembling anime video from frames...")
#     image = ffmpeg.input(f"{style_dir}/frame_%04d.jpg", framerate=fps).video
#     audio = ffmpeg.input(audio_file).audio
#     ffmpeg_command = ffmpeg.output(image, audio, output_video, vcodec="libx264", acodec='mp3', pix_fmt="yuv420p", shortest=None)
#     ffmpeg_command.run(overwrite_output=True)
#     print(f"✅ Anime video created: {output_video}")

# ==== Step 4: Main function to run the pipeline ====


if __name__ == "__main__":
    parser = ArgumentParser(description="AnimeGANv3 Video Style Transfer")

    # Define parameters
    parser.add_argument("-i", "--input", help="Input video file")
    parser.add_argument("-fd", "--frames_dir", help="Extract frames from video directory")
    parser.add_argument("-sd", "--style_dir", help="Stylize frames directory")
    # parser.add_argument("-s", "--style",default="AnimeGANv3_Hayao_36.onnx", help="Style model path")
    parser.add_argument("-d", "--device", default="cpu", help="Device to use (cpu or gpu cuda)")
    parser.add_argument("-f", "--fps", type=int, help="fps of output video")
    parser.add_argument("-it", "--img_type", default="jpg", help="Image frame type")
    parser.add_argument("-at", "--audio_type", default="mp3", help="Audio type")
    parser.add_argument("-o", "--output", help="Output video file")

    # Parse the arguments
    args = parser.parse_args()

    # check if input video is None
    if (args.input is None):
        args.input = fileOpenDialog("mp4")
        if (args.input is None or args.input == ""):
            print("Error: No video file selected. Exiting...")
            exit(0)
    # get input information type
    input_path = Path(args.input).parent.resolve()
    input_name = Path(args.input).stem
    input_ext = Path(args.input).suffix
    # print(f"Input video path: {args.input}")

    # select style model
    selected_style = libAnimeGANv3.getStyleModel(resources_path)
    print(f"Selected style model: {selected_style}")
    # remove old directories
    rmtree(temporary_dir, ignore_errors=True)
    makedirs(temporary_dir, exist_ok=True)
    makedirs(frame_dir, exist_ok=True)
    makedirs(style_dir, exist_ok=True)

    # check fps is None
    if (args.fps is None):
        args.fps = get_fps_fast(args.input)

    # check output video is None
    if (args.output is None):
        args.output = f"{input_path}\{input_name}_style{input_ext}"

    if (args.frames_dir is None):
        args.frames_dir = frame_dir
    if (args.style_dir is None):
        args.style_dir = style_dir
    frame_ = f"frame_%06d"
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
    processFrames = libFFMPEG.ExtractFrames(args.input, f"{args.frames_dir}\{frame_}.{args.img_type}", args.fps)
    processFrames.join()
    print("📽️ Extracting frames from done")

    print("🎶 Extracting audio from video...")
    processAudio = libFFMPEG.ExtractAudio(args.input, audio_file, args.audio_type)
    processAudio.join()
    if (processAudio.success == True):
        print("🎶 Extracting audio success")
    else:
        audio_file = None
        print("🎶 Extracting audio failed")

    # Create and start the thread
    processConvert = libAnimeGANv3.RunConvertImages(frame_dir, style_dir, selected_style, args.device)
    processConvert.join()

    # create video
    processCreateVideo = libFFMPEG.CreateVideoFromImages(f"{args.style_dir}\{frame_}.{args.img_type}", audio_file, args.output, round(args.fps))
    processCreateVideo.join()
    if (processCreateVideo.success == True):
        print("🎬  Video creation complete!")
    else:
        print(f'Error: {processCreateVideo.stderr}')

    # print(f"Creating video with command: {cmd}")

# ==== End of script ====
