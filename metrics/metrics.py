import os
import math
import tqdm

import pympi
import numpy as np
import pandas as pd

from pose_format import Pose
from pose_format.utils.generic import reduce_holistic

from ham2pose import compare_poses, mse, masked_mse, APE, masked_APE, fastdtw


def normalize_pose(pose: Pose) -> Pose:
    return pose.normalize(pose.header.normalization_info(
        p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"),
        p2=("POSE_LANDMARKS", "LEFT_SHOULDER")
    ))


def get_pose(pose_path: str):
    with open(pose_path, 'rb') as f:
        pose = Pose.read(f.read())

    if "WORLD_LANDMARKS" in [c.name for c in pose.header.components]:
        pose = pose.get_components(["POSE_LANDMARKS", "FACE_LANDMARKS", "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"])
    if "FACE_LANDMARKS" in [c.name for c in pose.header.components]:
        pose = reduce_holistic(pose)

    pose = normalize_pose(pose)

    return pose


def subpose(pose: Pose, start, end) -> Pose:
    pose = pose.get_components([c.name for c in pose.header.components])
    start = math.floor(start / 1000 * pose.body.fps)
    end = math.ceil(end / 1000 * pose.body.fps)
    pose.body.data = pose.body.data[start:end]
    pose.body.confidence = pose.body.confidence[start:end]
    return pose


text_path = './system_outputs/signsuisse_test/test.txt'
pose_dirs = {
    'ref': './system_outputs/signsuisse_test/ref/',
    # 'sockeye': './system_outputs/signsuisse_test/sockeye/',
    # 'sign_mt': './system_outputs/signsuisse_test/sign_mt/',
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
if __name__ == "__main__":
    save_every_n = 10
    
    for system in systems:
        print(system)

        with open(text_path) as file:
            output_csv_path = f'./metrics/metrics.{system}.csv'
            if os.path.exists(output_csv_path):
                entries = pd.read_csv(output_csv_path).to_dict('records')
                from_scratch = False
            else:
                entries = []
                from_scratch = True

            for index, line in tqdm.tqdm(enumerate(file)):
                # if index != 809:
                #     continue

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
                # pose_hyp_path = f"{pose_dirs[system]}{index}.pose"
                pose_hyp_path = f"{pose_dirs[system]}{index}.imputed.pose" if system == 'sockeye' else f"{pose_dirs[system]}{index}.raw.pose"
                pose_ref = get_pose(pose_ref_path)
                pose_hyp = get_pose(pose_hyp_path)

                # frame-level metrics from Ham2pose

                entry['nMSE'] = compare_poses(pose_hyp, pose_ref, distance_function=mse)
                entry['nAPE'] = compare_poses(pose_hyp, pose_ref, distance_function=APE)
                entry['DTW'] = compare_poses(pose_hyp, pose_ref, distance_function=fastdtw)
                entry['nDTW'] = compare_poses(pose_hyp, pose_ref, distance_function='nfastdtw')
                # if 'nMSE' not in entry:
                #     entry['nMSE'] = compare_poses(pose_hyp, pose_ref, distance_function=mse)
                # if 'nAPE' not in entry:
                #     entry['nAPE'] = compare_poses(pose_hyp, pose_ref, distance_function=APE)
                # if 'DTW' not in entry:
                #     entry['DTW'] = compare_poses(pose_hyp, pose_ref, distance_function=fastdtw)
                # if 'nDTW' not in entry:
                #     entry['nDTW'] = compare_poses(pose_hyp, pose_ref, distance_function='nfastdtw')

                # segment-level metrics based on sign language segmentation

                # seg_ref_path = f"{pose_dirs['ref']}{index}.eaf"
                # seg_hyp_path = f"{pose_dirs[system]}{index}.eaf"
                # seg_ref = pympi.Elan.Eaf(seg_ref_path).get_annotation_data_for_tier('SIGN')
                # seg_hyp = pympi.Elan.Eaf(seg_hyp_path).get_annotation_data_for_tier('SIGN')
                
                # if 'SegDiff' not in entry:
                #     entry['SegDiff'] = abs(len(seg_ref) - len(seg_hyp))
                
                # seg_pose_ref = [subpose(pose_ref, start, end) for start, end, _ in seg_ref]
                # seg_pose_hyp = [subpose(pose_hyp, start, end) for start, end, _ in seg_hyp]

                # if 'SegnAPERecall' not in entry or math.isnan(entry['SegnAPERecall']):
                #     if len(seg_pose_ref) == 0 or len(seg_pose_hyp) == 0:
                #         # fallback to frame-level when segmentation fails
                #         entry['SegnAPERecall'] = entry['nDTW']
                #     else:
                #         similarity_matrix = np.zeros((len(seg_pose_ref), len(seg_pose_hyp)))
                #         distance_function = lambda pose_ref, pose_hyp: compare_poses(pose_hyp, pose_ref, distance_function=APE)
                        
                #         for i, item_A in enumerate(seg_pose_ref):
                #             for j, item_B in enumerate(seg_pose_hyp):
                #                 similarity_matrix[i, j] = distance_function(item_A, item_B)
                        
                #         entry['SegnAPERecall'] = similarity_matrix.min(axis=1).mean()

                # if 'SegnDTWRecall' not in entry or math.isnan(entry['SegnDTWRecall']):
                #     if len(seg_pose_ref) == 0 or len(seg_pose_hyp) == 0:
                #         # fallback to frame-level when segmentation fails
                #         entry['SegnDTWRecall'] = entry['nDTW']
                #     else:
                #         similarity_matrix = np.zeros((len(seg_pose_ref), len(seg_pose_hyp)))
                #         distance_function = lambda pose_ref, pose_hyp: compare_poses(pose_hyp, pose_ref, distance_function='nfastdtw')
                        
                #         for i, item_A in enumerate(seg_pose_ref):
                #             for j, item_B in enumerate(seg_pose_hyp):
                #                 similarity_matrix[i, j] = distance_function(item_A, item_B)
                        
                #         # np.set_printoptions(linewidth=200)
                #         # print(similarity_matrix.shape)
                #         # print(similarity_matrix)
                #         # print(similarity_matrix.min(axis=1))

                #         # TODO: weighted average
                #         entry['SegnDTWRecall'] = similarity_matrix.min(axis=1).mean()

                #         # print(entry['SegnDTWRecall'])
                #         # exit()
                
                if from_scratch:
                    entries.append(entry)

                if index % save_every_n == (save_every_n - 1):
                    df = pd.DataFrame.from_dict(entries) 
                    df.to_csv(output_csv_path, index=False)