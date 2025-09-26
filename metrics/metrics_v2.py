import os
import math
import tqdm

import pympi
import numpy as np
import pandas as pd

from pose_format import Pose
from pose_format.utils.generic import reduce_holistic

from ham2pose import compare_poses, mse, masked_mse, APE, masked_APE, fastdtw

from pose_evaluation.metrics.base import BaseMetric
from pose_evaluation.metrics.distance_measure import AggregatedPowerDistance
from pose_evaluation.metrics.distance_metric import DistanceMetric
from pose_evaluation.metrics.test_distance_metric import get_poses
from pose_evaluation.utils.pose_utils import zero_pad_shorter_poses


def normalize_pose(pose: Pose) -> Pose:
    return pose.normalize(pose.header.normalization_info(
        p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
        p2=("POSE_LANDMARKS", "LEFT_SHOULDER")
    ))


def get_pose(pose_path: str):
    with open(pose_path, 'rb') as f:
        pose = Pose.read(f.read())

    if "WORLD_LANDMARKS" in [c.name for c in pose.header.components]:
        pose = pose.get_components([
            "POSE_LANDMARKS", "FACE_LANDMARKS",
            "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"
        ])
    if "FACE_LANDMARKS" in [c.name for c in pose.header.components]:
        pose = reduce_holistic(pose)

    pose = normalize_pose(pose)

    return pose


text_path = './system_outputs/signsuisse_test/test.txt'
pose_dirs = {
    'ref': './system_outputs/signsuisse_test/ref/',
    'sockeye': './system_outputs/signsuisse_test/sockeye/',
    'sign_mt': './system_outputs/signsuisse_test/sign_mt/',
    'sign_mt_v2': './system_outputs/signsuisse_test/sign_mt_v2/',
}
systems = list(pose_dirs.keys())[1:]
language_map = {
    'dsgs': {
        'source': 'deu',
        'target': 'sgg',
    },
    'lsf-ch': {
        'source': 'fra',
        'target': 'fsl',
    },
    'lis-ch': {
        'source': 'ita',
        'target': 'lse',
    },
}

metrics = [
    DistanceMetric(
        "MeanL2Score",
        AggregatedPowerDistance(order=2, aggregation_strategy="mean", default_distance=0),
    ),
]

if __name__ == "__main__":
    save_every_n = 10

    for system in systems:
        print(system)

        with open(text_path) as file:
            output_csv_path = f'./metrics/metrics.{system}.v2.csv'
            if os.path.exists(output_csv_path):
                entries = pd.read_csv(output_csv_path).to_dict('records')
                from_scratch = False
            else:
                entries = []
                from_scratch = True

            for index, line in tqdm.tqdm(enumerate(file)):
                sent = line.rstrip()
                language = sent.split(' ')[0][1:-1]

                entry = {
                    'data': 'signsuisse_test',
                    'system': system,
                    'example_id': index,
                    'source': language_map[language]['source'],
                    'target': language_map[language]['target'],
                } if from_scratch else entries[index]

                pose_ref_path = f"{pose_dirs['ref']}{index}.raw.pose"
                # Use imputed file for 'sockeye', otherwise use raw file
                pose_hyp_path = (
                    f"{pose_dirs[system]}{index}.imputed.pose"
                    if system == 'sockeye'
                    else f"{pose_dirs[system]}{index}.raw.pose"
                )
                pose_ref = get_pose(pose_ref_path)
                pose_hyp = get_pose(pose_hyp_path)

                poses = [pose_hyp, pose_ref]
                poses = zero_pad_shorter_poses(poses)

                for metric in metrics:
                    entry[metric.name] = metric.score(poses[0], poses[1])

                if from_scratch:
                    entries.append(entry)

                if index % save_every_n == (save_every_n - 1):
                    df = pd.DataFrame.from_dict(entries)
                    df.to_csv(output_csv_path, index=False)
