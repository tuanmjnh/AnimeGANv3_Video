# Install
pip install -r requirements.txt

# Run
py AnimeGANv3_Video.py -i "inputs\dance girl.mp4" -o ./outputs/anime_video.mp4 -f 30 -d gpu

# Build exe
pyinstaller --onefile AnimeGANv3_Video.py
pyinstaller -F --clean AnimeGANv3_Video.py
pyinstaller AnimeGANv3_Video.py --upx-dir=..\upx391w -y --onefile