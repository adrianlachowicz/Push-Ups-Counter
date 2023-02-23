import os
import cv2
import torch
import numpy as np
import mediapipe as mp
from model import Model
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
    parser.add_argument("--img-path", help="A path to a image.", type=str)
    parser.add_argument(
        "--classifier-path",
        help="A path to a classifier checkpoint.",
        type=str,
        default="src\models\epoch-91-model.pth",
    )
    args = parser.parse_args()
    return args


def load_image(img_path: str):
    """
    The function loads image and converts to the RGB format.

    Arguments:
        img_path (str) - A path to an image.

    Returns:
        img (np.array) - A loaded image.
    """

    img = cv2.imread(img_path)
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


def load_classifier(classifier_path: str):
    """
    The function loads classifier with weights from a specific checkpoint.

    Arguments:
        classifier_path (str) - A path to a classifier checkpoint.

    Returns
        model (nn.Module) - A classifier.
    """
    model = Model()
    model.load_state_dict(torch.load(classifier_path))

    return model


def extract_landmarks_and_distances(img: np.array, model: mp.solutions.pose.Pose):
    """
    The function extracts landmarks of shoulders and wirsts from a passed image and
    calculates distances between them in the Y axis..

    Arguments:
        img (np.array) - A image in the numpy array format.

    Returns:

    """

    results = model.process(img)

    # Variable naming convention
    #   first fragment:    l -> left   /   r -> right
    #   second fragment:   sh -> shoulder   /   el -> elbow   /   wr -> wrist
    #   third fragment:    x -> The X axis coordinate   /   y -> The Y axis coordinate

    l_sh_y = results.pose_landmarks.landmark[
        mp.solutions.pose.PoseLandmark.LEFT_SHOULDER
    ].y
    l_wr_y = results.pose_landmarks.landmark[
        mp.solutions.pose.PoseLandmark.LEFT_WRIST
    ].y

    r_sh_y = results.pose_landmarks.landmark[
        mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
    ].y
    r_wr_y = results.pose_landmarks.landmark[
        mp.solutions.pose.PoseLandmark.RIGHT_WRIST
    ].y

    l_distance = max(l_sh_y, l_wr_y) - min(l_sh_y, l_wr_y)
    r_distance = max(r_sh_y, r_wr_y) - min(r_sh_y, r_wr_y)

    return [l_distance, r_distance]


def preprocess_data(data: list):
    """
    The function preprocess data and prepares to be an input for classifier.

    Arguments:
        data (list) - Left and right side distances.

    Returns:
        output (torch.Tensor) - An input to classifier.
    """
    output = torch.tensor(data, dtype=torch.float32)

    return output


def predict(classifier, inputs: torch.tensor):
    """
    The function predicts labels of the image based on outputs from the Mediapipe model.

    Arguments:
        inputs (torch.tensor) - An input to a classifier.

    Returns:
        output (int) - 0 or 1, depends from up or down stage prediction.
        label (str) - A label of prediction.
    """

    target2label = {0: "Down stage", 1: "Up stage"}

    outputs = classifier(inputs)
    output = torch.softmax(outputs, dim=0).cpu().detach().numpy().argmax()
    label = target2label[output]

    return output, label


if __name__ == "__main__":
    args = parse_args()

    img_path = args.img_path
    classifier_path = args.classifier_path

    img = load_image(img_path)

    mediapipe_model = load_mediapipe_model()
    classifier_model = load_classifier(classifier_path)

    data = extract_landmarks_and_distances(img, mediapipe_model)
    data = preprocess_data(data)

    output, label = predict(classifier_model, data)

    print("Label: {}".format(label))
