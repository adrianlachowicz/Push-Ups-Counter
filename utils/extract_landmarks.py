import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def parse_args():
    """
    The function parses arguments passed in run command.

    Returns:
        arguments (Namespace) - Parsed arguments.
    """
    parser = ArgumentParser(description="The script extracts body landmarks from passed image.")
    parser.add_argument("--image-path", help="A path to a image.")
    args = parser.parse_args()
    return args


def load_image(image_path: str):
    """
    The function loads image from a passed path using the OpenCV library. 

    Arguments:
        image_path (str) - A specific image path.

    Returns:
        img (np.array) - A image.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img
    

def load_mediapipe_model():
    """
    The function loads Mediapipe model to detect human pose landmarks.

    Returns:
        pose (mp.solutions.pose.Pose) - The model with a specific configuration.
    """

    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(static_image_mode=True,
                    model_complexity=2,
                    enable_segmentation=True,
                    min_detection_confidence=0.5)

    return pose



if __name__ == "__main__":
    args = parse_args()

    image_path = args.image_path

    model = load_mediapipe_model()
    img = load_image(image_path)


    plt.imshow(img)
    plt.show()
