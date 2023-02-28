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
        description="The script runs counter on a passed video."
    )
    parser.add_argument("--video-path", help="A path to an video.", type=str)
    args = parser.parse_args()
    return args


def run_counter(video_path: str):
    """
    The function runs counter for an video passed by a path.
    It gets frame every 1 second and make predictions.
    Next it counts a number of push-ups.

    Arguments:
        video_path (str) - A path to an video
    """
    counter = 1

    cap = cv2.VideoCapture(video_path)
    count = 0
    counter += 1
    success = True

    while success:
        success,image = cap.read()
        print('read a new frame:',success)
        if count%30 == 0 :
            cv2.imwrite("/workspaces/Push-Ups-Counter/frames/{}.jpg".format(counter), image)
        count+=1


if __name__ == "__main__":
    args = parse_args()
    video_path = args.video_path

    run_counter(video_path)