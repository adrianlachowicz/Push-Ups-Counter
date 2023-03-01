curl -L "https://app.roboflow.com/ds/tKoZkOROVo?key=LVIqCNjsp9" > dataset.zip
mkdir ../data/
mkdir ../data/youtube_videos/
mkdir ../data/youtube_videos/1/

unzip dataset.zip -d ../data/youtube_videos/1/
rm dataset.zip