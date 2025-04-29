import os
import shutil
import libFFMPEG
from libIO import fileOpenDialog


# ==== Configuration ====
pic_form = ['.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG']
input_video = './inputs/video.mp4'
checkpoint = './checkpoint/paprika'
extract_dir = 'extracts'
frame_dir = f'{extract_dir}\\frames'
style_dir = f'{extract_dir}\\styles'
audio_file = f'{extract_dir}\\audio.mp3'
output_video = 'anime_video.mp4'
fps = 30  # Output video frame rate

# Example usage
if __name__ == '__main__':

    shutil.rmtree(extract_dir, ignore_errors=True)
    os.makedirs(extract_dir, exist_ok=True)
    os.makedirs(frame_dir, exist_ok=True)
    os.makedirs(style_dir, exist_ok=True)

    input_file = fileOpenDialog('mp4')
    if (input_file is None or input_file == ''):
        print('No video file selected. Exiting...')
        exit(0)
    print(f'Input file: {input_file}')
    video_info = libFFMPEG.getVideoInfo(input_file)

    # Run with callback
    fps = video_info.fps
    process = libFFMPEG.ExtractFrames(input_file, frame_dir, 'jpg', fps)
    # Optionally wait for the thread to complete
    process.join()
    print(F'Result: {process.success}')
