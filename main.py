import argparse
from enum import Enum
from typing import Iterator, List

import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

# from common.ball import BallTracker, BallAnnotator
from configs.basketball import BasketballCourtConfiguration

import warnings
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed to 'ensure_all_finite'")

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))

COURT_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/models/keypoint2.pt')
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/models/nba_detection.pt')

CONFIG = BasketballCourtConfiguration()

PLAYER_CLASS_ID = 'Player'
BALL_CLASS_ID = 0

STRIDE = 60

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
PADDING = 20

# VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
#     color=[sv.Color.from_hex(color) for color in CONFIG.colors],
#     text_color=sv.Color.from_hex('#FFFFFF'),
#     border_radius=5,
#     text_thickness=1,
#     text_scale=0.5,
#     text_padding=5,
# )
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)

class Mode(Enum):
    """
    Enum class representing different modes of operation for video analysis.
    """
    COURT_DETECTION = 'COURT_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    # BALL_DETECTION = 'BALL_DETECTION'


def run_court_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run court detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    court_detection_model = YOLO(COURT_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = court_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        CLASS_TO_INDEX = {
            "Center Line": 0,
            "Left Paint": 1,
            "Left Three Point": 2,
            "Right Paint": 3,
            "Right Three Point": 4,
        }

        flattened_keypoints = []
        filtered_labels = []
        filtered_colors = []

        num_kps_per_object = 5

        for obj_idx in range(keypoints.xy.shape[0]):
            class_name = keypoints.data['class_name'][obj_idx]
            if class_name not in CLASS_TO_INDEX:
                continue

            base_idx = CLASS_TO_INDEX[class_name] * num_kps_per_object
            labels_slice = CONFIG.labels[base_idx:base_idx + num_kps_per_object]
            colors_slice = CONFIG.colors[base_idx:base_idx + num_kps_per_object]

            for kp_idx, (x, y) in enumerate(keypoints.xy[obj_idx]):
                if x == 0 and y == 0:
                    continue
                flattened_keypoints.append([x, y])
                filtered_labels.append(labels_slice[kp_idx])
                filtered_colors.append(colors_slice[kp_idx])
        
        VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
            color=[sv.Color.from_hex(color) for color in filtered_colors],
            text_color=sv.Color.from_hex('#FFFFFF'),
            border_radius=5,
            text_thickness=1,
            text_scale=0.5,
            text_padding=5,
        )

        flattened_keypoints = np.array(flattened_keypoints, dtype=np.float32)
        flattened_keypoints += np.array([PADDING, PADDING], dtype=np.float32)
        flattened_keypoints = np.array([flattened_keypoints])
        flattened_keypoints = sv.KeyPoints(xy=flattened_keypoints)

        padded_frame = cv2.copyMakeBorder(
            frame.copy(),
            top=PADDING,
            bottom=PADDING,
            left=PADDING,
            right=PADDING,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            padded_frame, flattened_keypoints, filtered_labels)

        yield annotated_frame


def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        
        detections.xyxy = detections.xyxy + np.array([PADDING, PADDING, PADDING, PADDING], dtype=np.float32)

        annotated_frame = cv2.copyMakeBorder(
            frame.copy(),
            top=PADDING,
            bottom=PADDING,
            left=PADDING,
            right=PADDING,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

        # annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        # annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        # only annotate players
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections[detections.data['class_name'] == PLAYER_CLASS_ID])
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections[detections.data['class_name'] == PLAYER_CLASS_ID])

        yield annotated_frame


# def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
#     """
#     Run ball detection on a video and yield annotated frames.

#     Args:
#         source_video_path (str): Path to the source video.
#         device (str): Device to run the model on (e.g., 'cpu', 'cuda').

#     Yields:
#         Iterator[np.ndarray]: Iterator over annotated frames.
#     """
#     ball_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
#     frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    
#     ball_tracker = BallTracker(buffer_size=20)
#     ball_annotator = BallAnnotator(radius=6, buffer_size=10)

#     def callback(image_slice: np.ndarray) -> sv.Detections:
#         result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
#         detections = sv.Detections.from_ultralytics(result)
#         detections =  detections[detections.class_id == BALL_CLASS_ID]
#         return detections[detections.confidence > 0.5]

#     slicer = sv.InferenceSlicer(
#         callback=callback,
#         slice_wh=(640, 640),
#         overlap_wh=[0.3,0.3],
#         overlap_ratio_wh=None,
#     )

#     for frame in frame_generator:
#         detections = slicer(frame).with_nms(threshold=0.25)
#         detections = ball_tracker.update(detections)
#         annotated_frame = frame.copy()
#         annotated_frame = ball_annotator.annotate(annotated_frame, detections)
#         yield annotated_frame


def main(source_video_path: str, target_video_path: str, device: str, mode: Mode) -> None:

    if mode == Mode.COURT_DETECTION:
        frame_generator = run_court_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path, device=device)
    # elif mode == Mode.BALL_DETECTION:
    #     frame_generator = run_ball_detection(
    #         source_video_path=source_video_path, device=device)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    padded_resolution = (
        video_info.resolution_wh[0] + 2 * PADDING,
        video_info.resolution_wh[1] + 2 * PADDING
    )
    updated_video_info = sv.VideoInfo(
        fps=video_info.fps,
        width=padded_resolution[0],
        height=padded_resolution[1]
    )
    with sv.VideoSink(target_video_path, updated_video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)

            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, default ='output.mp4')
    parser.add_argument('--device', type=str, default='cpu')
    # parser.add_argument('--mode', type=Mode, default=Mode.RADAR)
    parser.add_argument('--mode', type=Mode, default=Mode.BALL_DETECTION)
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode
    )
