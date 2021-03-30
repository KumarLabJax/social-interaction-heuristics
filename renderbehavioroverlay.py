import argparse
import cv2
import h5py
import imageio
import numpy as np
import os
from pathlib import Path
import urllib.parse as urlparse
import yaml

import gensocialstats
import rendervidoverlay
import socialutil

NOSE_INDEX = 0
LEFT_EAR_INDEX = 1
RIGHT_EAR_INDEX = 2
BASE_NECK_INDEX = 3
LEFT_FRONT_PAW_INDEX = 4
RIGHT_FRONT_PAW_INDEX = 5
CENTER_SPINE_INDEX = 6
LEFT_REAR_PAW_INDEX = 7
RIGHT_REAR_PAW_INDEX = 8
BASE_TAIL_INDEX = 9
MID_TAIL_INDEX = 10
TIP_TAIL_INDEX = 11

CONNECTED_SEGMENTS = [
        [LEFT_FRONT_PAW_INDEX, CENTER_SPINE_INDEX, RIGHT_FRONT_PAW_INDEX],
        [LEFT_REAR_PAW_INDEX, BASE_TAIL_INDEX, RIGHT_REAR_PAW_INDEX],
        [
            NOSE_INDEX, BASE_NECK_INDEX, CENTER_SPINE_INDEX,
            BASE_TAIL_INDEX, MID_TAIL_INDEX, TIP_TAIL_INDEX,
        ],
]

OVERLAY_COLOR = (255, 102, 0)

# colors from color brewer: https://colorbrewer2.org/?type=qualitative&scheme=Paired&n=9
QUALITATIVE_COLORS = [
    (166,206,227),
    (31,120,180),
    (178,223,138),
    (51,160,44),
    (251,154,153),
    (227,26,28),
    (253,191,111),
    (255,127,0),
    (202,178,214),
]

FRAME_BUFFER = 30 * 3
VIDEO_ANNOTATION_PADDING_PX = 112
TEXT_HEIGHT_PX = 22
FRAME_NUM_VERTICAL_OFFSET_PX = 90

# share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar'
# python -u ~/projects/social-interaction/renderbehavioroverlay.py \
#   --batch-file ~/projects/rotta-data/rotta-labeler-comp_2020_12_18.txt \
#   --video-root-dir "${share_root}" \
#   --behavior-root-dirs rotta-labeler-comp_2020_12_18_arojit \
#                        rotta-labeler-comp_2020_12_18_yehya \
#   --behavior Approach \
#   --annotator-names AM YB \
#   --out-dir behavior-out-vids-arojit-yehya \
#   --allow-missing-video

# share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar'
# python -u ~/projects/social-interaction/renderbehavioroverlay.py \
#   --batch-file ~/projects/rotta-data/rotta-labeler-comp_2020_12_18.txt \
#   --video-root-dir "${share_root}" \
#   --behavior-root-dirs rotta-labeler-comp_2020_12_18_arojit \
#                        rotta-labeler-comp_2020_12_18_yehya \
#   --behavior Leave \
#   --annotator-names AM YB \
#   --out-dir behavior-out-vids-arojit-yehya \
#   --allow-missing-video

# share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar'
# python -u ~/projects/social-interaction/renderbehavioroverlay.py \
#   --batch-file ~/projects/rotta-data/rotta-labeler-comp_2020_12_18.txt \
#   --video-root-dir "${share_root}" \
#   --behavior-root-dirs rotta-labeler-comp_2020_12_18_arojit \
#                        rotta-labeler-comp_2020_12_18_yehya \
#   --behavior Chase \
#   --annotator-names AM YB \
#   --out-dir behavior-out-vids-arojit-yehya \
#   --allow-missing-video

# share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar'
# python -u ~/projects/social-interaction/renderbehavioroverlay.py \
#   --batch-file ~/projects/rotta-data/rotta-labeler-comp_2021-01-27_amelie/rotta-labeler-comp_2020_12_18.txt \
#   --video-root-dir "${share_root}" \
#   --behavior-root-dirs rotta-labeler-comp_2021-01-27_amelie \
#                        rotta-labeler-comp_2021-01-27_arojit \
#                        rotta-labeler-comp_2021-01-27_yehya \
#   --behavior Leave \
#   --behavior-aliases Leave Leave_A_P Leave2 \
#   --annotator-names AB AM YB \
#   --out-dir rotta-labeler-comp_2021-01-27_vid-out3 \
#   --allow-missing-video
#
# share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar'
# python -u ~/projects/social-interaction/renderbehavioroverlay.py \
#   --batch-file ~/projects/rotta-data/rotta-labeler-comp_2021-01-27_amelie/rotta-labeler-comp_2020_12_18.txt \
#   --video-root-dir "${share_root}" \
#   --behavior-root-dirs rotta-labeler-comp_2021-01-27_arojit \
#                        rotta-labeler-comp_2021-01-27_yehya \
#   --behavior Leave \
#   --behavior-aliases Leave_A_P Leave2 \
#   --annotator-names AM YB \
#   --out-dir rotta-labeler-comp_2021-01-27_vid-out2 \
#   --allow-missing-video

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch-file',
        help='the batch file',
        required=True,
    )

    parser.add_argument(
        '--video-root-dir',
        help='input root directory for videos',
        required=True,
    )

    parser.add_argument(
        '--behavior-root-dirs',
        nargs='+',
        help='input root directorys for behavior data (HDF5 files)',
        required=True,
    )

    parser.add_argument(
        '--behavior',
        help='behavior to process',
        required=True,
    )

    parser.add_argument(
        '--behavior-aliases',
        nargs='+',
        help='if the labelers used names that don\'t match the given'
             ' behavior you can use this option to list what users'
             ' specified. This option should either be missing or'
             ' it should have the same length as "--behavior-root-dirs"',
        default=[],
    )

    parser.add_argument(
        '--annotator-names',
        nargs='+',
        help='names to match up with each annotator',
        required=True,
    )

    parser.add_argument(
        '--out-dir',
        help='output directory for behavior clips',
        required=True,
    )

    parser.add_argument(
        '--allow-missing-video',
        help='allow missing videos with warning',
        action='store_true',
    )

    args = parser.parse_args()

    assert len(args.annotator_names) == len(args.behavior_root_dirs)

    video_root_dir = Path(args.video_root_dir)
    behavior_root_dirs = [Path(d) for d in args.behavior_root_dirs]
    out_dir = Path(args.out_dir)

    exclude_points = set()
    exclude_points.add(rendervidoverlay.LEFT_FRONT_PAW_INDEX)
    exclude_points.add(rendervidoverlay.RIGHT_FRONT_PAW_INDEX)
    exclude_points.add(rendervidoverlay.LEFT_EAR_INDEX)
    exclude_points.add(rendervidoverlay.RIGHT_EAR_INDEX)

    with open(args.batch_file, 'r') as batch_file:
        for net_id in batch_file:
            net_id = net_id.strip()
            print(f"Working on {net_id}")

            #rotta-labeler-comp_2020_12_18_amelie/NV16-UCSD/2019-11-07/3901455_2019-11-08_11-00-00_behavior/v1/Approach/3901455_2019-11-08_11-00-00.h5
            vid_path = video_root_dir / net_id
            pose_path = vid_path.parent / (vid_path.stem + '_pose_est_v3.h5')

            if not vid_path.exists():
                print(f'Skipping because "{vid_path}" does not exist')
                continue

            pose_exists = pose_path.exists()
            # if not pose_exists:
            #     print(f'Skipping because "{pose_path}" does not exist')
            #     continue

            if pose_exists:
                with h5py.File(pose_path, 'r') as pose_data:
                    all_points = pose_data['poseest/points'][:]
                    all_instance_count = pose_data['poseest/instance_count'][:]
                    all_instance_track_id = pose_data['poseest/instance_track_id'][:]
                    all_points_mask = pose_data['poseest/confidence'][:] > 0

            def find_track_for_frame(frame_index, candidate_ids):

                frame_instance_count = all_instance_count[frame_index]
                for frame_track_id in all_instance_track_id[frame_index, :frame_instance_count]:
                    if int(frame_track_id) in candidate_ids:
                        # print('FOUND IT', frame_track_id)
                        return int(frame_track_id)

                return None

            collapsed_pred_class = []
            all_pred_class = []
            all_id_to_tracks = []
            for i, behavior_root_dir in enumerate(behavior_root_dirs):

                curr_behavior = args.behavior
                if args.behavior_aliases:
                    curr_behavior = args.behavior_aliases[i]

                h5_path = behavior_root_dir / net_id
                h5_path = h5_path.parent / (h5_path.stem + '_behavior') / 'v1' / curr_behavior / (h5_path.stem + '.h5')

                # print(f'"{net_id}": {vid_path.exists()}, {h5_path}, {h5_path.exists()}')

                with h5py.File(h5_path, 'r') as h5_data:
                    pred_class = h5_data['predictions/predicted_class'][:] == 1
                    pred_any_true = pred_class.any(axis=0)

                    all_pred_class.append(pred_class)
                    collapsed_pred_class.append(pred_any_true)

                    id_to_track = h5_data['predictions/identity_to_track'][:]
                    id_count, frame_count = id_to_track.shape
                    id_to_tracks = []
                    for curr_id in range(id_count):
                        track_id_set = set()
                        for curr_track_id in id_to_track[curr_id, :]:
                            if curr_track_id >= 0:
                                track_id_set.add(int(curr_track_id))
                        id_to_tracks.append(track_id_set)

                    all_id_to_tracks.append(id_to_tracks)

            collapsed_pred_class = np.stack(collapsed_pred_class)
            annotator_count, frame_count = collapsed_pred_class.shape

            out_video_path = out_dir / args.behavior / net_id
            out_video_path.parent.mkdir(parents=True, exist_ok=True)
            with imageio.get_reader(vid_path) as video_reader, \
                 imageio.get_writer(out_video_path, fps=30) as video_writer:
                for frame_index, frame in enumerate(video_reader):

                    active_track_to_annotators = dict()

                    if frame_index % 10000 == 0:
                        print(f"Processed frame {frame_index + 1}")

                    win_start = max(0, frame_index - FRAME_BUFFER)
                    win_stop = min(frame_count, frame_index + FRAME_BUFFER + 1)

                    keep_frame = collapsed_pred_class[:, win_start:win_stop].sum() >= 1

                    if keep_frame:
                        frame_row_count, frame_col_count, frame_color_count = frame.shape
                        annotation_start_row = frame_row_count - VIDEO_ANNOTATION_PADDING_PX
                        cv2.putText(
                            frame,
                            'Frame #: {}'.format(frame_index + 1),
                            (5, annotation_start_row + FRAME_NUM_VERTICAL_OFFSET_PX),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1.0,
                            OVERLAY_COLOR,
                        )

                        track_id_to_true_annos = dict()

                        for anno_index in range(annotator_count):
                            # print('anno_index:', anno_index)
                            behavior_active = collapsed_pred_class[anno_index, frame_index] == 1
                            anno_x = 5
                            anno_y = annotation_start_row + FRAME_NUM_VERTICAL_OFFSET_PX - ((anno_index + 1) * 30)
                            cv2.putText(
                                frame,
                                f"{args.behavior} - {args.annotator_names[anno_index]}",
                                (anno_x + TEXT_HEIGHT_PX * 2, anno_y),
                                cv2.FONT_HERSHEY_COMPLEX,
                                1.0,
                                OVERLAY_COLOR,
                            )
                            cv2.rectangle(
                                frame,
                                (anno_x, anno_y - TEXT_HEIGHT_PX),
                                (anno_x + TEXT_HEIGHT_PX, anno_y),
                                OVERLAY_COLOR,
                                cv2.FILLED if behavior_active else 1,
                            )

                            if pose_exists:
                                if behavior_active:
                                    curr_anno_all_preds = all_pred_class[anno_index]
                                    id_count, frame_count = curr_anno_all_preds.shape
                                    curr_anno_id_to_tracks = all_id_to_tracks[anno_index]
                                    for curr_id in range(id_count):
                                        if curr_anno_all_preds[curr_id, frame_index]:
                                            # the current annotator has called true for behavior
                                            # for the current ID at the current frame. We still
                                            # need to locate the corresponding track
                                            active_track = find_track_for_frame(
                                                frame_index,
                                                all_id_to_tracks[anno_index][curr_id])
                                            if active_track is not None:
                                                if active_track in active_track_to_annotators:
                                                    active_track_to_annotators[active_track].append(anno_index)
                                                else:
                                                    active_track_to_annotators[active_track] = [anno_index]

                                for track, anno_ids in active_track_to_annotators.items():
                                    # get track location
                                    frame_instance_count = all_instance_count[frame_index]
                                    curr_frame_track_ids = [
                                        int(tid)
                                        for tid in all_instance_track_id[frame_index, :frame_instance_count]
                                    ]
                                    frame_track_index = curr_frame_track_ids.index(track)
                                    if all_points_mask[frame_index, frame_track_index, CENTER_SPINE_INDEX]:
                                        center_spine_y, center_spine_x = all_points[frame_index, frame_track_index, CENTER_SPINE_INDEX, :]
                                        cv2.circle(
                                            frame,
                                            (center_spine_x, center_spine_y),
                                            5,
                                            OVERLAY_COLOR,
                                            cv2.FILLED,
                                        )
                                        cv2.putText(
                                            frame,
                                            ', '.join(sorted([args.annotator_names[i] for i in anno_ids])),
                                            (center_spine_x + 6, center_spine_y + 6),
                                            cv2.FONT_HERSHEY_COMPLEX,
                                            1.0,
                                            OVERLAY_COLOR,
                                        )

                        video_writer.append_data(frame)


if __name__ == '__main__':
    main()
