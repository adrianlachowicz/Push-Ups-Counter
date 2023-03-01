import os
import cv2
from argparse import ArgumentParser


def parse_args():
    """
    The function parses arguments passed in run command.

    Returns:
        arguments (Namespace) - Parsed arguments.
    """
    parser = ArgumentParser(
        description="The script splits video to frames with specific time spacing."
    )
    parser.add_argument("--video-path", type=str, help="A path to a video.")
    parser.add_argument(
        "--frames-directory",
        type=str,
        help="A path to a directory, where frames should be saved.",
    )
    parser.add_argument(
        "--spacing", type=int, help="Spacing between frames.", default=10
    )
    args = parser.parse_args()

    return args


def split_video_to_frames(video_path: str, frames_dir: str, spacing: int) -> None:
    """
    The function splits video to frames with specific spaces between frames and saves in a specific directory.

    Arguments:
        video_path (str) - A path to a video.
        frames_dir (str) - A path to a directory, where frames should be saved.
        spacing (int) - Spacing between frames.
    """

    capture = cv2.VideoCapture(video_path)
    frame_counter = 0

    while True:
        success, frame = capture.read()

        if success:
            if frame_counter % spacing == 0:
                frame_filename = str(frame_counter) + ".jpg"
                frame_path = os.path.join(frames_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
        else:
            break

        frame_counter += 1

    capture.release()


if __name__ == "__main__":
    args = parse_args()

    video_path = args.video_path
    frames_directory_path = args.frames_directory
    spacing = args.spacing

    split_video_to_frames(video_path, frames_directory_path, spacing)
