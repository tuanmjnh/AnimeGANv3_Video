import os
import shutil
import libFFMPEG
from libIO import fileOpenDialog


# ==== Configuration ====
pic_form = ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG']
input_video = "./inputs/video.mp4"
checkpoint = "./checkpoint/paprika"
extract_dir = "extracts"
frame_dir = f"{extract_dir}\\frames"
style_dir = f"{extract_dir}\\styles"
audio_file = f"{extract_dir}\\audio.mp3"
output_video = "anime_video.mp4"
fps = 30  # Output video frame rate

# Example usage
if __name__ == "__main__":
    
    shutil.rmtree(extract_dir, ignore_errors=True)
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(style_dir, exist_ok=True)

    input_file = fileOpenDialog("mp4")
    if (input_file is None or input_file == ""):
        print("No video file selected. Exiting...")
        exit(0)
    video_info = libFFMPEG.getVideoInfo(input_file)
    print(video_info.duration, video_info.width, video_info.height, video_info.bit_rate, video_info.total_frames, video_info.r_frame_rate, video_info.fps)
    # process = SubProcessThread(libFFMPEG.cmdExtractAudio(input_file, audio_file, audio_format="mp3"), options=SimpleNamespace(text=True, check=True))
    process = libFFMPEG.ExtractAudio(input_file, audio_file, audio_format="mp3")
    process.join()
    print(process.success)
    # if process.success:
    #     print(f"Action success!: {process.stdout}")
    # else:
    #     print(f"Action failure!: {process.stderr}")

    print(f"Input file: {input_file}")
    # Run with callback
    # thread = extract_audio(input_file, audio_file, audio_format="mp3", callback=handle_result)
    # Optionally wait for the thread to complete
    # thread.join()
