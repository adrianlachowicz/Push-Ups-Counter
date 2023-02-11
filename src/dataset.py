import torch
import pandas as pd
from torch.utils.data import Dataset


class PushUpsDataset(Dataset):
    """
    The dataset. It loads data from a .csv file, preprocess inputs, and returns in an appropriate format.
    The output data consist of:
        - Inputs for model:
            - LEFT_SHOULDER_ELBOW_DISTANCE
            - LEFT_SHOULDER_WRIST_DISTANCE
            - RIGHT_SHOULDER_ELBOW_DISTANCE
            - RIGHT_SHOULDER_WRIST_DISTANCE
        - A target for a classification output
            - Label (0 as a down stage and 1 as an up stage)

    Arguments:
        csv_path (str) - A path to a .csv file, which contains data about landmarks
                         on specific image and distances between them.
        dir_path (str) - A path to a directory, which contains images.
    """

    def __init__(self, csv_path: str, dir_path: str):
        self.df = pd.read_csv(csv_path)
        self.dir_path = dir_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.loc[item]

        left_shoulder_elbow_distance = row["LEFT_SHOULDER_ELBOW_DISTANCE"]
        left_shoulder_wrist_distance = row["LEFT_SHOULDER_WRIST_DISTANCE"]
        right_shoulder_elbow_distance = row["RIGHT_SHOULDER_ELBOW_DISTANCE"]
        right_shoulder_wrist_distance = row["RIGHT_SHOULDER_WRIST_DISTANCE"]
        label = row["Label"]

        inputs = [
            left_shoulder_elbow_distance,
            left_shoulder_wrist_distance,
            right_shoulder_elbow_distance,
            right_shoulder_wrist_distance,
        ]
        inputs = torch.tensor(inputs, dtype=torch.float32)

        targets = label
        targets = torch.tensor(targets, dtype=torch.int64)

        return inputs, targets
