# Zephyr Basketball Homography

## install

```bash
pip install -r requirements.txt
```

## models

- bball-keypoint (data/models/keypoint2.pt)
- nba-game-detection-gbbjt (data/models/nba_detection.pt)

## modes

- `COURT_DETECTION`
- `PLAYER_DETECTION`
- `PLAYER_TRACKING`
- `TEAM_CLASSIFICATION`
- `RADAR` (default)

```bash
python3 main.py \
--source_video_path data/input/filename.mp4 \
--target_video_path data/output/filename.mp4 \
--device mps --mode COURT_DETECTION
```

## credits

https://github.com/roboflow/sports/tree/main
