from json import loads
from libThread import SubProcessThread
from types import SimpleNamespace


def getVideoInfo(input_video):
    class VideoInfo:
        def __init__(self, duration, width, height, bit_rate, total_frames, r_frame_rate, fps):
            self.duration = duration
            self.width = width
            self.height = height
            self.bit_rate = bit_rate
            self.total_frames = total_frames
            self.r_frame_rate = r_frame_rate
            self.fps = fps
    # Get the total frame count and original fps of the video
    cmd = [
        'ffprobe',
        '-v', 'error',              # Hide unnecessary logs
        '-select_streams', 'v:0',   # Select first video stream
        '-show_entries', 'stream=nb_frames,r_frame_rate,duration,width,height,bit_rate',  # Get frame number and FPS
        '-of', 'json',              # Output format is JSON
        input_video
    ]
    # result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    process = SubProcessThread(cmd, options=SimpleNamespace(stdoutPipe=True, stderrPipe=True, text=True, check=True))
    process.start()
    process.join()
    # Parse JSON output
    info = loads(process.stdout)
    stream = info.get('streams', [{}])[0]

    r_frame_rate = stream.get('r_frame_rate', '0/1')
    # Get FPS (r_frame_rate is in fraction form, e.g., '30/1')
    num, denom = map(int, r_frame_rate.split('/'))
    fps = num / denom if denom != 0 else 0

    video_info = VideoInfo(
        float(stream.get('duration', 0)),
        int(stream.get('width', 0)),
        int(stream.get('height', 0)),
        int(stream.get('bit_rate', 0)),
        int(stream.get('nb_frames', 0)),
        r_frame_rate,
        fps)
    return video_info


# Audio
def cmdExtractAudio(input_video, output_audio, audio_format='mp3'):
    # FFmpeg command to extract audio
    return [
        'ffmpeg',
        '-i', input_video,          # Input file
        '-vn',                      # No video
        '-acodec', audio_format,    # Audio codec (mp3, pcm_s16le for wav)
        '-y',                       # Overwrite output file if exists
        output_audio,               # Output file
    ]


def ExtractAudio(input_video, output_audio, audio_format='mp3'):
    process = SubProcessThread(cmdExtractAudio(input_video, output_audio, audio_format), options=SimpleNamespace(text=True, check=True))
    process.start()
    return process


# Frames
def cmdExtractFrames(input_video, output_dir, fps=None):
    # output_pattern = os.path.join(output_dir, f'frame_%06d.{output_format}')
    return [
        'ffmpeg',
        '-i', input_video,          # Input file
        '-vf', f'fps={fps}',        # Use source frame rate
        '-y',                       # Overwrite output files
        '-hide_banner',
        output_dir                  # Output pattern (e.g., frame_000001.png)
    ]


def ExtractFrames(input_video, output_dir, fps=None):
    process = SubProcessThread(cmdExtractFrames(input_video, output_dir, fps), options=SimpleNamespace(text=True, check=True))
    process.start()
    return process


# Video
def cmdCreateVideoFromImages(images_pattern, audio_file, output_video, fps):
    cmd = [
        'ffmpeg',
        '-i', images_pattern,              # Input images
        # '-i', audio_file,                # Input audio
        '-c:v', 'libx264',                 # Video codec
        '-framerate', fps,                 # Set frame rate (you can change 30 to what you want)
        '-r', fps,                         # Output frame rate
        '-pix_fmt', 'yuv420p',             # Pixel format (important for compatibility)
        '-shortest',                       # Stop encoding when the shortest input ends (useful to match audio length)
        '-y',                       # Overwrite output files
        '-hide_banner',
        output_video
    ]
    if (audio_file is not None):
        cmd.insert(3, '-i')
        cmd.insert(4, audio_file)
    print(cmd)
    return cmd


def CreateVideoFromImages(images_pattern, audio_file, output_video, fps):
    process = SubProcessThread(cmdCreateVideoFromImages(images_pattern, audio_file, output_video, str(fps)), options=SimpleNamespace(text=True, check=True))
    process.start()
    return process
