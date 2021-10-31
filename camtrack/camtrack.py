#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
from numpy.core.numeric import indices
import sortednp as snp

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    TriangulationParameters,
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    rodrigues_and_translation_to_view_mat3x4
)

import cv2

def calc_new_view_mat(points3d, old_corners, new_corners, intrinsic_mat, prev_view_mat, id_qualities):
    _, (possible_idx1, possible_idx2) = snp.intersect(points3d[1], new_corners.ids.reshape(-1), indices=True)

    _, rvec, tvec, inliers = cv2.solvePnPRansac(points3d[0][possible_idx1], new_corners.points[possible_idx2], intrinsic_mat, None, reprojectionError=15)
    view_mat = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)

    new_bad_ids = np.setdiff1d(points3d[1][possible_idx1], points3d[1][possible_idx1][inliers])

    if new_bad_ids.shape[0] > 0:
        id_qualities = np.pad(id_qualities, (0, max(0, new_bad_ids.max() - id_qualities.shape[0] + 1)))
        id_qualities[new_bad_ids] += 1

    # if points3d[1][possible_idx1].shape[0] > 20:
    #     bad_ids = snp.merge(bad_ids, new_bad_ids, duplicates=snp.DROP)
    triang_params = TriangulationParameters(max_reprojection_error=15, min_triangulation_angle_deg=0, min_depth=0)

    correspondences = build_correspondences(old_corners, new_corners)
    _points3d = triangulate_correspondences(correspondences, prev_view_mat, view_mat, intrinsic_mat, triang_params)
    
    new_bad_ids = np.setdiff1d(correspondences[0], _points3d[1])
    if new_bad_ids.shape[0] > 0:
        id_qualities = np.pad(id_qualities, (0, max(0, new_bad_ids.max() - id_qualities.shape[0] + 1)))
        id_qualities[new_bad_ids] += 1


    new_ids, (inds1, inds2) = snp.merge(_points3d[1], points3d[1], indices=True, duplicates=snp.DROP)
    
    new_points3d = np.zeros((new_ids.shape[0], 3))
    new_points3d[inds1] = _points3d[0]  
    new_points3d[inds2] = points3d[0]
    
    bad_ids = np.argwhere(id_qualities > 1).reshape(-1)

    _, (bad_idx, _) = snp.intersect(new_ids, bad_ids, indices=True)

    new_points3d = np.delete(new_points3d, bad_idx, axis=0)
    new_ids = np.delete(new_ids, bad_idx, axis=0)

    return view_mat, (new_points3d, new_ids), id_qualities

def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    correspondences = build_correspondences(corner_storage[known_view_1[0]], corner_storage[known_view_2[0]])
    triang_params = TriangulationParameters(max_reprojection_error=15, min_triangulation_angle_deg=0, min_depth=0)
    points3d = triangulate_correspondences(correspondences, pose_to_view_mat3x4(known_view_1[1]), pose_to_view_mat3x4(known_view_2[1]), intrinsic_mat, triang_params)
    id_qualities = np.array([], dtype=int)
    bad_ids = np.array([], dtype=int)

    frame_count = len(corner_storage)
    view_mats = [None]*frame_count
    view_mats[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])

    min_dist = 5
    max_dist = 9

    not_processed_idx = set(range(frame_count))
    not_processed_idx.remove(known_view_1[0])
    not_processed_idx.remove(known_view_2[0])

    while not_processed_idx:
        for i in not_processed_idx.copy():
            if view_mats[i] is None:
                for j in range(i+min_dist, min(frame_count, i+max_dist)):
                    if view_mats[j] is not None:
                        try:
                            view_mats[i], points3d, id_qualities  = calc_new_view_mat(points3d, corner_storage[j], corner_storage[i], intrinsic_mat, view_mats[j], id_qualities)
                            not_processed_idx.remove(i)
                            break
                        except cv2.error:
                            points3d = triangulate_correspondences(correspondences, pose_to_view_mat3x4(known_view_1[1]), pose_to_view_mat3x4(known_view_2[1]), intrinsic_mat, triang_params)
                            id_qualities = id_qualities = np.array([], dtype=int)
                if view_mats[i] is None:
                    for j in range(i-min_dist, max(0, i-max_dist), -1):
                        if view_mats[j] is not None:
                            try:
                                view_mats[i], points3d, id_qualities  = calc_new_view_mat(points3d, corner_storage[j], corner_storage[i], intrinsic_mat, view_mats[j], id_qualities)
                                not_processed_idx.remove(i)
                                break
                            except cv2.error:
                                points3d = triangulate_correspondences(correspondences, pose_to_view_mat3x4(known_view_1[1]), pose_to_view_mat3x4(known_view_2[1]), intrinsic_mat, triang_params)
                                id_qualities = id_qualities = np.array([], dtype=int)

    point_cloud_builder = PointCloudBuilder(points3d[1],
                                            points3d[0])

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
