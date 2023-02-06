import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def parse_args():
    """
    The function parses arguments passed in run command.

    Returns:
        arguments (Namespace) - Parsed arguments.
    """
    parser = ArgumentParser(
        description="The script extracts body landmarks from passed image."
    )
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

    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5,
    )

    return pose


def extract_landmarks_from_image(img: np.array, model: mp.solutions.pose.Pose):
    """
    The function extracts landmarks from image using the Mediapipe model and returns appropriate coordinates.

    Arguments:
        img (np.array) - A image
        model (mp.solutions.pose.Pose) - The Mediapipe model, which predicts pose landmarks

    Returns:
        coordinates_df (pd.DataFrame) - A DataFrame, which contains coordinates in separate columns.
    """

    results = model.process(img)

    # Variable naming convention
    #   first fragment:    l -> left   /   r -> right
    #   second fragment:   sh -> shoulder   /   el -> elbow   /   wr -> wrist
    #   third fragment:    x -> The X axis coordinate   /   y -> The Y axis coordinate

    # Left shoulder landmarks
    l_sh = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    l_sh_x = l_sh.x
    l_sh_y = l_sh.y

    # Left elbow landmarks
    l_el = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
    l_el_x = l_el.x
    l_el_y = l_el.y

    # Left wrist landmarks
    l_wr = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
    l_wr_x = l_wr.x
    l_wr_y = l_wr.y

    # Right shoulder landmarks
    r_sh = results.pose_landmarks.landmark[
        mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
    ]
    r_sh_x = r_sh.x
    r_sh_y = r_sh.y

    # Right elbow landmarks
    r_el = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
    r_el_x = r_el.x
    r_el_y = r_el.y

    # Right wrist landmarks
    r_wr = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
    r_wr_x = r_wr.x
    r_wr_y = r_wr.y

    data = {
        "LEFT_SHOULDER_X": l_sh_x,
        "LEFT_SHOULDER_Y": l_sh_y,
        "LEFT_ELBOW_X": l_el_x,
        "LEFT_ELBOW_Y": l_el_y,
        "LEFT_WRIST_X": l_wr_x,
        "LEFT_WRIST_Y": l_wr_y,
        "RIGHT_SHOULDER_X": l_sh_x,
        "RIGHT_SHOULDER_Y": r_sh_y,
        "RIGHT_ELBOW_X": r_el_x,
        "RIGHT_ELBOW_Y": r_el_y,
        "RIGHT_WRIST_X": r_wr_x,
        "RIGHT_WRIST_Y": r_wr_y,
    }

    coordinates_df = pd.DataFrame(data, index=[0])

    return coordinates_df


if __name__ == "__main__":
    args = parse_args()

    image_path = args.image_path

    model = load_mediapipe_model()
    img = load_image(image_path)

    extract_landmarks_from_image(img, model)
