#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
from numpy.core.numeric import indices
import sortednp as snp

from corners import CornerStorage, FrameCorners
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
    rodrigues_and_translation_to_view_mat3x4,
    _remove_correspondences_with_ids,
    eye3x4
)

from random import shuffle

import cv2

triang_params = TriangulationParameters(
    max_reprojection_error=3, min_triangulation_angle_deg=1.0, min_depth=0.11)

def calc_new_view_mat(points3d, old_corners, new_corners, intrinsic_mat, prev_view_mat, bad_ids):
    possible_ids = snp.intersect(
        points3d[1], new_corners.ids.reshape(-1))

    bad_ids, (bad_idx, _) = snp.intersect(possible_ids, bad_ids, indices=True)

    good_ids = np.delete(possible_ids, bad_idx)
    _, (_, possible_idx1) = snp.intersect(
        good_ids, points3d[1], indices=True)
    _, (_, possible_idx2) = snp.intersect(
        good_ids, new_corners.ids.reshape(-1), indices=True)

    succesful, rvec, tvec, inliers = cv2.solvePnPRansac(
        points3d[0][possible_idx1],
        new_corners.points[possible_idx2],
        intrinsic_mat,
        distCoeffs=None,
        reprojectionError=7,
        confidence=0.999)

    
    if not succesful or (np.any(inliers) and len(inliers) < 5):
        raise cv2.error

    view_mat = rodrigues_and_translation_to_view_mat3x4(rvec, tvec)

    new_bad_ids = np.setdiff1d(
        points3d[1][possible_idx1], points3d[1][possible_idx1][inliers])

    # if points3d[1][possible_idx1].shape[0] > 20:
    #     bad_ids = snp.merge(bad_ids, new_bad_ids, duplicates=snp.DROP)
        
    correspondences = build_correspondences(old_corners, new_corners, new_bad_ids)
    _points3d = triangulate_correspondences(
        correspondences, prev_view_mat, view_mat, intrinsic_mat, triang_params)

    _new_bad_ids = np.setdiff1d(correspondences[0], _points3d[1])
    new_bad_ids = snp.merge(new_bad_ids, _new_bad_ids, duplicates=snp.DROP)
    new_bad_ids = snp.merge(new_bad_ids, bad_ids, duplicates=snp.DROP)

    new_ids, (inds1, inds2) = snp.merge(
        _points3d[1], points3d[1], indices=True, duplicates=snp.DROP)

    new_points3d = np.zeros((new_ids.shape[0], 3))
    new_points3d[inds1] = _points3d[0]
    new_points3d[inds2] = points3d[0]

    # _, (bad_idx, _) = snp.intersect(new_ids, bad_ids, indices=True)

    # new_points3d = np.delete(new_points3d, bad_idx, axis=0)
    # new_ids = np.delete(new_ids, bad_idx, axis=0)

    return view_mat, (new_points3d, new_ids), new_bad_ids, len(_points3d[1])


def build_views(corners1: FrameCorners, corners2: FrameCorners, intrinsic_mat: np.ndarray, params):
    correspondences = build_correspondences(corners1, corners2)
    if len(correspondences[0]) < 200:
        return 0, None, None
    essential_mat, inliers = cv2.findEssentialMat(correspondences[1], correspondences[2], intrinsic_mat, method=cv2.RANSAC, prob=params["confidence"]) 
    correspondences = _remove_correspondences_with_ids(
            correspondences, np.argwhere(inliers.flatten() == 0).astype(np.int64))

    _, hom_inliers = cv2.findHomography(correspondences[1], correspondences[2], **params)
    if np.sum(hom_inliers) / np.sum(inliers) > 0.7:
        return 0, None, None

    r1, r2, t = cv2.decomposeEssentialMat(essential_mat)

    best_cnt = 0
    best_view = None

    for rot in (r1, r2):
        for translation in (t, -t):
            view1 = eye3x4()
            view2 = np.hstack((rot, translation))

            _, ids, _ = triangulate_correspondences(
                correspondences, view1, view2, intrinsic_mat, triang_params)

            if best_cnt < len(ids):
                best_view = view2
                best_cnt = len(ids)

    return best_cnt, eye3x4(), best_view


def initiallize(
    corner_storage: CornerStorage,
    intrinsic_mat: np.ndarray
):
    CONFIDENCE = 0.999
    MAX_ITERS = 10 ** 4
    THRESHOLD_PX = 1.0
    params_opencv = dict(
        method=cv2.RANSAC,
        ransacReprojThreshold=THRESHOLD_PX,
        confidence=CONFIDENCE,
        maxIters=MAX_ITERS
    )

    frame_count = len(corner_storage)
    min_dist = 2
    max_dist = 20
    min_cnt = 1000

    best_frames = None
    best_cnt = 0


    for i in range(frame_count):
        for j in range(i+min_dist, min(frame_count, i+max_dist)):    
            cnt, view1, view2 = build_views(corner_storage[i], corner_storage[j], intrinsic_mat, params_opencv)
            if cnt > best_cnt:
                best_frames = ((i, view1), (j, view2))
                best_cnt = cnt


        for j in range(i-min_dist, max(0, i-max_dist), -1):
            cnt, view1, view2 = build_views(corner_storage[i], corner_storage[j], intrinsic_mat, params_opencv)
            if cnt > best_cnt:
                best_frames = ((i, view1), (j, view2))
                best_cnt = cnt

        print(f"Frame {i}/{frame_count}, best inlier count: {best_cnt}")
        if best_cnt > min_cnt:
            break

    print(best_cnt)

    return best_frames


from tqdm import tqdm

def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    frame_count = len(corner_storage)
    view_mats = [None]*frame_count
    
    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = initiallize(corner_storage, intrinsic_mat)
        view_mats[known_view_1[0]] = known_view_1[1]
        view_mats[known_view_2[0]] = known_view_2[1]
    else:
        view_mats[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
        view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])

    correspondences = build_correspondences(
        corner_storage[known_view_1[0]], corner_storage[known_view_2[0]])
    
    points3d = triangulate_correspondences(correspondences, 
        view_mats[known_view_1[0]], view_mats[known_view_2[0]], intrinsic_mat, triang_params)
    id_qualities = np.array([], dtype=int)

    min_dist = 1
    max_dist = 70

    not_processed_idx = set(range(frame_count))
    not_processed_idx.remove(known_view_1[0])
    not_processed_idx.remove(known_view_2[0])

    min_quality = 1

    frame_errors = [0]*len(view_mats)
    best_cnts = [0]*len(view_mats)
    best_cnts[known_view_1[0]] = 1e9
    best_cnts[known_view_2[0]] = 1e9
    bad_ids = [np.empty(0, dtype=int)]*len(view_mats)

    while not_processed_idx:
        processing_order = sorted(list(not_processed_idx))
        print(processing_order)
        print(best_cnts)
        for i in tqdm(processing_order):
            best_res = None
            other_idx = list(range(i+min_dist, min(frame_count, i+max_dist))) + list(range( max(0, i-max_dist),i-min_dist))
            for j in other_idx:
                if view_mats[j] is not None and best_cnts[j] > 200:
                    try:
                        res = calc_new_view_mat(
                            points3d, corner_storage[j], corner_storage[i], intrinsic_mat, view_mats[j], bad_ids[j])
                        if not best_res or res[3] > best_res[3]:
                            best_res = res
                    except cv2.error:
                        # id_qualities = np.clip(id_qualities - 1, 0, 1000)
                        pass

            if best_res:
                best_cnts[i] = best_res[3]

            if best_res and best_res[3] > 50:
                view_mats[i], points3d, bad_ids[i], _ = best_res
                not_processed_idx.remove(i)

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
