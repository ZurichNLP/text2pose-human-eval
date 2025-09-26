import os
import json
import tqdm
import argparse

import numpy as np

from pose_format import Pose
from pose_format.utils.generic import correct_wrists, reduce_holistic
from pose_format.utils.openpose import hand_colors
from pose_format.pose_visualizer import PoseVisualizer


def normalize_pose(pose: Pose) -> Pose:
    return pose.normalize(pose.header.normalization_info(
        p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
        p2=("POSE_LANDMARKS", "LEFT_SHOULDER")
    ))


def get_pose(pose_path: str):
    # Load video frames
    with open(pose_path, 'rb') as f:
        pose = Pose.read(f.read())

    pose = normalize_pose(pose)

    if "WORLD_LANDMARKS" in [c.name for c in pose.header.components]:
        pose = pose.get_components(["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"])
    if "FACE_LANDMARKS" in [c.name for c in pose.header.components]:
        pose = reduce_holistic(pose)

    pose = correct_wrists(pose)

    # Scale the newly created pose
    new_width = 500
    shift = 1.25
    shift_vec = np.full(shape=(pose.body.data.shape[-1]), fill_value=shift, dtype=np.float32)
    pose.body.data = (pose.body.data + shift_vec) * new_width
    pose.header.dimensions.height = pose.header.dimensions.width = int(new_width * shift * 2)

    return pose


def render_pose(pose, output_path: str):
    for component in pose.header.components:
        if component.name in ['LEFT_HAND_LANDMARKS', 'RIGHT_HAND_LANDMARKS']:
            component.colors = hand_colors
            
    visualizer = PoseVisualizer(pose)
    visualizer.save_video(output_path, visualizer.draw())


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

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default=None, help="Specify the language to process, if not specified, process all.")
args = parser.parse_args()

text_path = './system_outputs/signsuisse_test/test.txt'
pose_dirs = {
    # 'ref': './system_outputs/signsuisse_test/ref/',
    # 'sockeye': './system_outputs/signsuisse_test/sockeye/',
    # 'sign_mt': './system_outputs/signsuisse_test/sign_mt/',
    'sign_mt_v2': './system_outputs/signsuisse_test/sign_mt_v2/',
}

with open(text_path) as file:
    for index, line in tqdm.tqdm(enumerate(file)):
        sent = line.rstrip()

        # placeholders for these not translated by sign.mt v2
        if index not in [253, 459, 931, 60, 986]:
            continue

        # Check if the language is specified or not
        if args.language is None or sent.startswith(f'<{args.language}>'):
            text = ' '.join(sent.split(' ')[1:])

            for system, pose_dir in pose_dirs.items():
                pose_path = f'{pose_dir}{index}.pose' if system == 'sockeye' else f'{pose_dir}{index}.raw.pose'

                if os.path.exists(pose_path):
                    video_path = pose_path.replace('.pose', '.mp4') if system == 'sockeye' else pose_path.replace('.raw.pose', '.mp4')

                    if not os.path.exists(video_path):
                        pose = get_pose(pose_path)
                        render_pose(pose, video_path)
                else:
                    print(f'WARNING: pose {pose_path} does not exist')
