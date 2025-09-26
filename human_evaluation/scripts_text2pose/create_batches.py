import os
import json
import tqdm

import numpy as np

from pose_format import Pose
from pose_format.utils.generic import correct_wrists, reduce_holistic
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

    # pose = pose.get_components(["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"])

    pose = normalize_pose(pose)

    if "FACE_LANDMARKS" in pose.header.components:
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
    visualizer = PoseVisualizer(pose)
    visualizer.save_video(output_path, visualizer.draw())


language_map = {
    'dsgs': {
        'source': 'deu',
        'taget': 'sgg',
    },
    'lsf': {
        'source': 'fra',
        'taget': 'fsl',
    },
    'lis': {
        'source': 'ita',
        'taget': 'lse',
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default='dsgs')
args = parser.parse_args()

text_path = '/shares/iict-sp2.ebling.cl.uzh/zifjia/easier-continuous-translation/data_old/common/dsgs/signsuisse/test.txt'
pose_dirs = {
    'ref': '/shares/iict-sp2.ebling.cl.uzh/zifjia/easier-continuous-translation/models/signsuisse/test_ref/',
    'sockeye': '/shares/iict-sp2.ebling.cl.uzh/zifjia/easier-continuous-translation/models/signsuisse/test/',
}
segments = {
    'ref': [],
    'sockeye': [],
}

document_size = 10
document_id_prefix = 1000000
with open(text_path) as file:
    for index, line in tqdm.tqdm(enumerate(file)):
        sent = line.rstrip()

        if sent.startswith(f'<{arg.language}>'):
            text = ' '.join(sent.split(' ')[1:])

            for system, pose_dir in pose_dirs.items():
                pose_path = f'{pose_dir}{index}.pose'
                video_path = pose_path.replace('.pose', '.mp4')

                if not os.path.exists(video_path):
                    pose = get_pose(pose_path)
                    render_pose(pose, video_path)

                video_path_remote = f'https://pub.cl.uzh.ch/projects/iict/signsuisse_test/{system}/{index}.mp4'
                new_id = len(segments[system])
                document_id = int(new_id / document_size)

                segment = {
                    "_block": document_id,
                    "_item": new_id,
                    "itemID": index,
                    "itemType": "REF" if system == 'ref' else "TGT",
                    "sourceID": "signsuisse_test",
                    "sourceText": text,
                    "targetID": system,
                    "targetText": video_path_remote,
                }
                segments[system].append(segment)

for system, items in segments.items():
    batch_size = 100
    batches = []
    current_batch_items = []

    for item in items:
        current_batch_items.append(item)

        if len(current_batch_items) == batch_size:
            batch = {
                "items": current_batch_items,
                "task": {
                    "batchNo": len(batches) + 1,
                    "batchSize": batch_size,
                    "randomSeed": 1111,
                    "requiredAnnotations": 1,
                    "sourceLanguage": language_map[arg.language]['source'],
                    "targetLanguage": language_map[arg.language]['target'],
                },
            }

            batches.append(batch)
            current_batch_items = []

    output_path = f'./human_evaluation/batches_text2pose/batches.text2pose.signsuisse.{language_map[arg.language]['source']}-{language_map[arg.language]['target']}.{system}.json'
    with open(output_path, 'w') as fp:
        json.dump(batches, fp, indent=2)
