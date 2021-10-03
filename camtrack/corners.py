#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    
    max_corners = 3000

    params = {
        "qualityLevel": 0.03,
        "blockSize": 10,
        "minDistance": 10
    }

    # TODO
    image_0 = frame_sequence[0]
    
    corners = cv2.goodFeaturesToTrack(image_0, maxCorners=max_corners, **params)

    corners = FrameCorners(
        np.arange(corners.shape[0]),
        corners,
        np.array([10]*corners.shape[0])
    )

    builder.set_corners_at_frame(0, corners)
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        new_corners, st, err = cv2.calcOpticalFlowPyrLK((image_0 * 255).astype(np.uint8), (image_1 * 255).astype(np.uint8), corners.points, None, winSize=(30,30), maxLevel=2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                              10, 0.03))

        inds = (st == 1).reshape(-1)
        good_corners = new_corners[inds]

        mask = np.ones(image_1.shape).astype(np.uint8)
        for corner in good_corners:
            cv2.circle(mask, (int(corner[0]), int(corner[1])), 100, 0)

        ids = corners.ids[inds]

        if sum(inds) < max_corners:
            extra_corners = cv2.goodFeaturesToTrack(image_1, maxCorners=max_corners-sum(inds), mask=mask, **params)
            starting_id = max(ids) + 1
            ids = np.concatenate((ids, np.arange(starting_id, starting_id + extra_corners.shape[0]).reshape(-1, 1)), axis=0)
            good_corners = np.concatenate((good_corners.reshape((-1, 1, 2)), extra_corners), axis=0)


        corners = FrameCorners(
            ids,
            good_corners,
            np.array([10]*good_corners.shape[0])
        )
        


        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
