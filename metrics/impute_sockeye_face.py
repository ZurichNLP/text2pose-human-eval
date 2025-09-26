from tqdm import tqdm
from functools import lru_cache

import numpy as np
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.utils.generic import pose_normalization_info

from pose_anonymization.data import load_mean_and_std, load_pose_header


def normalize_pose_size(pose: Pose):
    new_width = 500
    shift = 1.25
    shift_vec = np.full(shape=(pose.body.data.shape[-1]), fill_value=shift, dtype=np.float32)
    pose.body.data = (pose.body.data + shift_vec) * new_width
    pose.header.dimensions.height = pose.header.dimensions.width = int(new_width * shift * 2)


def reduce_face(pose: Pose):
    # To avoid installing mediapipe, we just hardcode the face contours given the above code
    face_contours = [
        '0', '7', '10', '13', '14', '17', '21', '33', '37', '39', '40', '46', '52', '53', '54', '55', '58', '61', '63',
        '65', '66', '67', '70', '78', '80', '81', '82', '84', '87', '88', '91', '93', '95', '103', '105', '107', '109',
        '127', '132', '133', '136', '144', '145', '146', '148', '149', '150', '152', '153', '154', '155', '157', '158',
        '159', '160', '161', '162', '163', '172', '173', '176', '178', '181', '185', '191', '234', '246', '249', '251',
        '263', '267', '269', '270', '276', '282', '283', '284', '285', '288', '291', '293', '295', '296', '297', '300',
        '308', '310', '311', '312', '314', '317', '318', '321', '323', '324', '332', '334', '336', '338', '356', '361',
        '362', '365', '373', '374', '375', '377', '378', '379', '380', '381', '382', '384', '385', '386', '387', '388',
        '389', '390', '397', '398', '400', '402', '405', '409', '415', '454', '466'
    ]

    components = [c.name for c in pose.header.components if c.name != 'POSE_WORLD_LANDMARKS']
    return pose.get_components(components, {
        "FACE_LANDMARKS": face_contours,
    })


@lru_cache(maxsize=1)
def get_mean_appearance():
    mean, _ = load_mean_and_std()

    data = mean.reshape((1, 1, -1, 3)) * 1000
    confidence = np.ones((1, 1, len(mean)))
    body = NumPyPoseBody(fps=1, data=data, confidence=confidence)
    pose = Pose(header=load_pose_header(), body=body)

    return pose

pose_dir = './system_outputs/signsuisse_test/sockeye/'

for i in tqdm(range(1000)):
    pose_path = f'{pose_dir}/{i}.pose'
    pose_imputed_path = f'{pose_dir}/{i}.imputed.pose'

    with open(pose_path, 'rb') as f:
        pose = Pose.read(f.read())
        pose = pose.normalize(pose_normalization_info(pose.header))

        # get an average full pose
        mean_pose = get_mean_appearance()
        mean_pose = mean_pose.get_components(['POSE_LANDMARKS', 'FACE_LANDMARKS', 'LEFT_HAND_LANDMARKS', 'RIGHT_HAND_LANDMARKS'])
        mean_pose = mean_pose.normalize(pose_normalization_info(mean_pose.header))

        mean_pose.body.data = np.repeat(mean_pose.body.data, pose.body.data.shape[0], axis=0)
        mean_pose.body.confidence = np.repeat(mean_pose.body.confidence, pose.body.data.shape[0], axis=0)
        mean_pose.body.fps = pose.body.fps

        # replace the average pose with existing points
        points = [
            mean_pose.header._get_point_index(c.name, n)
            for c in pose.header.components
            for n in c.points
        ]
        mean_pose.body.data[:, :, points] = pose.body.data

        # impute the face by nose position
        nose_position_mean = mean_pose.body.data[:, :, mean_pose.header._get_point_index('FACE_LANDMARKS', '4')]
        nose_position_hyp = mean_pose.body.data[:, :, mean_pose.header._get_point_index('POSE_LANDMARKS', 'NOSE')]
        nose_position_diff = nose_position_hyp - nose_position_mean

        face_points = [
            mean_pose.header._get_point_index(c.name, n)
            for c in mean_pose.header.components if c.name == 'FACE_LANDMARKS'
            for n in c.points
        ]
        mean_pose.body.data[:, :, face_points] = mean_pose.body.data[:, :, face_points] + np.expand_dims(nose_position_diff, axis=2)

        normalize_pose_size(mean_pose)

        with open(pose_imputed_path, "wb") as f:
            mean_pose.write(f)