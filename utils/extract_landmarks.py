import os
import cv2
import math
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
from argparse import ArgumentParser


def parse_args():
    """
    The function parses arguments passed in run command.

    Returns:
        arguments (Namespace) - Parsed arguments.
    """
    parser = ArgumentParser(
        description="The script extracts body landmarks from passed images by .csv file from the Roboflow service."
    )
    parser.add_argument("--csv-path", help="A path to a .csv file.")
    parser.add_argument(
        "--dir-path", help="A path to a directory, which contains images."
    )
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


def extract_landmarks_from_image(
    img: np.array, img_path: str, model: mp.solutions.pose.Pose
):
    """
    The function extracts landmarks from image using the Mediapipe model and returns appropriate coordinates.

    Arguments:
        img (np.array) - A image.
        img_path (str) - A path to an image (only for inserting into a data row).
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

    # Calculate distances between landmarks
    # Left side body
    l_shoulder_elbow_distance = calculate_distance_between_points(
        l_sh_x, l_sh_y, l_el_x, l_el_y
    )
    l_shoulder_wrist_distance = calculate_distance_between_points(
        l_sh_x, l_sh_y, l_wr_x, l_wr_y
    )

    # Right side body
    r_shoulder_elbow_distance = calculate_distance_between_points(
        r_sh_x, r_sh_y, r_el_x, r_el_y
    )
    r_shoulder_wrist_distance = calculate_distance_between_points(
        r_sh_x, r_sh_y, r_wr_x, r_wr_y
    )

    data = {
        "IMG_PATH": img_path,
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
        "LEFT_SHOULDER_ELBOW_DISTANCE": l_shoulder_elbow_distance,
        "LEFT_SHOULDER_WRIST_DISTANCE": l_shoulder_wrist_distance,
        "RIGHT_SHOULDER_ELBOW_DISTANCE": r_shoulder_elbow_distance,
        "RIGHT_SHOULDER_WRIST_DISTANCE": r_shoulder_wrist_distance,
    }

    coordinates_df = pd.DataFrame(data, index=[0])

    return coordinates_df


def calculate_distance_between_points(
    point_a_x: float, point_a_y: float, point_b_x: float, point_b_y: float
) -> float:
    """
    The function calculates distance between two points.

    Arguments:
        point_a_x (float) - A X axis coordinate of the A point.
        point_a_y (float) - A Y axis coordinate of the A point.
        point_b_x (float) - A X axis coordinate of the B point.
        point_b_y (float) - A Y axis coordinate of the A point.

    Returns:
        distance (float) - A distance between the A point and the B point.
    """
    distance = math.sqrt(
        math.pow(point_a_x - point_b_x, 2) + math.pow(point_a_y - point_b_y, 2)
    )

    return distance


def extract_landmarks(csv_path: str, dir_path: str):
    """
    The function extracts landmarks from all images in a specific directory.
    Next, when the Mediapipe model end processing images, it saves outputs as .csv file.

    Arguments:
        csv_path (str) -  A path to a .csv file.
        dir_path (str) - A path to a directory, which contains images.
    """
    data = pd.DataFrame(
        columns=[
            "IMG_PATH",
            "LEFT_SHOULDER_X",
            "LEFT_SHOULDER_Y",
            "LEFT_ELBOW_X",
            "LEFT_ELBOW_Y",
            "LEFT_WRIST_X",
            "LEFT_WRIST_Y",
            "RIGHT_SHOULDER_X",
            "RIGHT_SHOULDER_Y",
            "RIGHT_ELBOW_X",
            "RIGHT_ELBOW_Y",
            "RIGHT_WRIST_X",
            "RIGHT_WRIST_Y",
            "LEFT_SHOULDER_ELBOW_DISTANCE",
            "LEFT_SHOULDER_WRIST_DISTANCE",
            "RIGHT_SHOULDER_ELBOW_DISTANCE",
            "RIGHT_SHOULDER_WRIST_DISTANCE",
        ]
    )

    model = load_mediapipe_model()
    df = pd.read_csv(csv_path)
    files = df["filename"].values

    labels_vec = df[[" push_down", " push_up"]].values
    labels = []

    # Labels for Cross Entropy Loss
    #   Push down -> 0
    #   Push up   -> 1

    for vec in labels_vec:
        labels.append(np.argmax(vec))

    labels = pd.Series(labels, name="Label")

    current_file = ""

    for file in tqdm(files):
        try:
            current_file = file
            file_path = os.path.join(dir_path, file)
            img = load_image(file_path)

            row = extract_landmarks_from_image(img, file_path, model)
            data = pd.concat([data, row], ignore_index=True)
        except:
            print("Error on processing image: {}".format(current_file))

    data = pd.concat([data, labels], axis="columns")

    data.to_csv("./data.csv", index=False)


if __name__ == "__main__":
    args = parse_args()
    csv_path = args.csv_path
    dir_path = args.dir_path

    extract_landmarks(csv_path, dir_path)
