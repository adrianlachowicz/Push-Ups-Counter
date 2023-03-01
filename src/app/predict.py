import cv2
import torch
import numpy as np
import mediapipe as mp
from PIL import Image
from model import Model
from config import *


def load_models():
    """
    The function loads the Mediapipe model and the classifier with specific weights from a checkpoint.

    Returns:
        pose (mp.solutions.pose.Pose) - The Mediapipe model for predicting human pose landmarks.
        classifier (nn.Module) - The classifier
    """

    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5,
    )

    classifier = Model()
    classifier.load_state_dict(
        torch.load(
            CLASSIFIER_CHECKPOINT,
            map_location=torch.device(DEVICE),
        )
    )

    return pose, classifier


def extract_landmarks_and_distances(img: np.array, model: mp.solutions.pose.Pose):
    """
    The function extracts landmarks of shoulders and wirsts from a passed image and
    calculates distances between them in the Y axis..

    Arguments:
        img (np.array) - A image in the numpy array format.

    Returns:
        l_distance (float) - A distance between left shoulder and left wrist in the Y axis.
        r_distance (float) - A distance between right shoulder and right wrist in the Y axis.

        -1 (int) - If the Mediapipe model can't extract landmarks -> Return -1.
    """

    results = model.process(img)

    # Variable naming convention
    #   first fragment:    l -> left   /   r -> right
    #   second fragment:   sh -> shoulder   /   el -> elbow   /   wr -> wrist
    #   third fragment:    x -> The X axis coordinate   /   y -> The Y axis coordinate
    try:
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
    except:
        return -1


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


def predict(img: np.array):
    """
    The function predicts label for a passed image.

    Arguments:
        img (np.array) - A image.

    Returns:
        label (int) - A label (0 for down stage and 1 for up stage).
    """

    target2label = {0: "Down stage", 1: "Up stage"}

    mp_model, classifier_model = load_models()

    # Extract data from image
    data = extract_landmarks_and_distances(img, mp_model)
    if data != -1:
        data = preprocess_data(data)

        # Predict label using classifier
        outputs = classifier_model(data)
        probs, labels = torch.topk(torch.softmax(outputs, dim=0), k=1)
        prob = probs.item()
        label = labels.item()

        return label
    else:
        return -1
