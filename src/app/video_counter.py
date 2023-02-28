import os
import cv2
from predict import predict
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
    valid_labels = [0, 1]
    last_valid_label = -1
    previous_label = -1
    push_ups_counter = 0

    
    stage_up = False
    stage_down = False
    last_stage_up = False

    cap = cv2.VideoCapture(video_path)
    frame_counter = 0

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            cv2.imshow('Frame', frame)
          
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            # Make predictions every 1 second
            if frame_counter % 30 == 0:
                label = predict(frame)

                if label in valid_labels:
                    last_valid_label = label
                
                if last_valid_label == 0:
                    stage_down = True
                elif last_valid_label == 1:
                    stage_up = True

                if previous_label == 0 and last_valid_label == 1:
                    last_stage_up = True

                if stage_up and stage_down and last_stage_up:
                    push_ups_counter += 1
                    stage_up = False
                    stage_down = False
                    last_stage_up = False

                previous_label = last_valid_label
                print(push_ups_counter)

            frame_counter += 1
            
        else:
            break
    


if __name__ == "__main__":
    args = parse_args()
    video_path = args.video_path

    run_counter(video_path)