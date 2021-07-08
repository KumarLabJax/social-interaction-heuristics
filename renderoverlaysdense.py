import argparse
import cv2
import h5py
import imageio
import itertools
import numpy as np
import os
from pathlib import Path
import urllib.parse as urlparse
import yaml
import json

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

# FRAME_BUFFER = 30 * 3
# VIDEO_ANNOTATION_PADDING_PX = 112
# TEXT_HEIGHT_PX = 22
# FRAME_NUM_VERTICAL_OFFSET_PX = 90


class BehaviorInterval(object):

    def __init__(
            self,
            start_frame, stop_frame_exclu,
            behavior_label,
            track1_id, track2_id):

        self.start_frame = start_frame
        self.stop_frame_exclu = stop_frame_exclu
        self.behavior_label = behavior_label
        self.track1_id = track1_id
        self.track2_id = track2_id


def main():

    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        '--batch-file',
        help='the batch file',
    )

    group.add_argument(
        '--video-file',
        help='the single video to process',
    )

    parser.add_argument(
        '--video-root-dir',
        help='input root directory for videos',
        default='.',
    )

    parser.add_argument(
        '--jabs-annotation',
        nargs=3,
        metavar=('rootdir', 'behaviorname', 'overlaylabel'),
        action='append',
        default=[],
    )

    parser.add_argument(
        '--heuristic-behavior',
        nargs=3,
        metavar=('yamlfile', 'behaviorname', 'overlaylabel'),
        action='append',
        default=[],
    )

    parser.add_argument(
        '--jabs-classification',
        nargs=3,
        metavar=('rootdir', 'behaviorname', 'overlaylabel'),
        action='append',
        default=[],
    )

    # parser.add_argument(
    #     '--annotator-names',
    #     nargs='+',
    #     help='names to match up with each annotator',
    #     required=True,
    # )

    parser.add_argument(
        '--out-dir',
        help='output directory for behavior clips',
        required=True,
    )

    # parser.add_argument(
    #     '--allow-missing-video',
    #     help='allow missing videos with warning',
    #     action='store_true',
    # )

    args = parser.parse_args()

    # assert len(args.annotator_names) == len(args.behavior_root_dirs)

    video_root_dir = Path(args.video_root_dir)
    # behavior_root_dirs = [Path(d) for d in args.behavior_root_dirs]
    out_dir = Path(args.out_dir)

    exclude_points = set()
    exclude_points.add(rendervidoverlay.LEFT_FRONT_PAW_INDEX)
    exclude_points.add(rendervidoverlay.RIGHT_FRONT_PAW_INDEX)
    exclude_points.add(rendervidoverlay.LEFT_EAR_INDEX)
    exclude_points.add(rendervidoverlay.RIGHT_EAR_INDEX)

    def gen_net_ids():
        if args.video_file is not None:
            yield args.video_file
        else:
            with open(args.batch_file, 'r') as batch_file:
                for net_id in batch_file:
                    yield net_id.strip()

    net_ids = list(gen_net_ids())
    all_behaviors = dict()
    all_track_to_id_mapping = dict()
    for net_id in net_ids:
        all_behaviors[net_id] = []
        all_track_to_id_mapping[net_id] = dict()
    # all_identity_to_track = dict()

    for net_id in net_ids:

        net_id_root, _ = os.path.splitext(net_id)

        for rootdir, behaviorname, overlaylabel in args.jabs_classification:
            jabspath = os.path.join(
                rootdir,
                net_id_root + '_behavior', 'v1',
                behaviorname,
                net_id_root + '.h5')

            with h5py.File(jabspath, 'r') as h5_data:
                pred_class = h5_data['predictions/predicted_class'][:]
                pred_prob = h5_data['predictions/probabilities'][:]
                identity_to_track = h5_data['predictions/identity_to_track'][:]
                valid_behavior = np.logical_and(pred_class == 1, pred_prob >= 0)

            # map tracklet IDs to identities
            id_count, frame_count = identity_to_track.shape
            for id_index in range(id_count):
                for frame_index in range(frame_count):
                    tracklet_id = identity_to_track[id_index, frame_index]
                    if tracklet_id != -1:
                        if tracklet_id in all_track_to_id_mapping[net_id]:
                            assert all_track_to_id_mapping[net_id][tracklet_id] == id_index
                        else:
                            all_track_to_id_mapping[net_id][tracklet_id] = id_index

            # all_identity_to_track[net_id] = identity_to_track

            for id_index in range(id_count):
                curr_grp_start_index = 0
                behavior_grps = itertools.groupby(valid_behavior[id_index, :])
                for k, grp in behavior_grps:
                    grp_len = sum(1 for _ in grp)
                    if k:
                        bi = BehaviorInterval(
                            curr_grp_start_index,
                            curr_grp_start_index + grp_len,
                            overlaylabel,
                            id_index,
                            None,
                        )
                        all_behaviors[net_id].append(bi)

                    curr_grp_start_index += grp_len


        for rootdir, behaviorname, overlaylabel in args.jabs_annotation:
            annopath = os.path.join(rootdir, net_id_root + '.json')
            with open(annopath, 'r') as json_file:
                gt_doc = json.load(json_file)
                vid_id = gt_doc['file']
                assert vid_id == net_id
                # print('vid_id:', vid_id)

                for id_str, label_dict in gt_doc['labels'].items():
                    for beh_name, beh_intervals in label_dict.items():
                        if beh_name == behaviorname:
                            # print('===', id_str, beh_name, '===')
                            for beh_interval in beh_intervals:
                                if beh_interval['present']:
                                    # print(beh_interval)
                                    bi = BehaviorInterval(
                                        beh_interval['start'],
                                        beh_interval['end'],
                                        overlaylabel,
                                        int(id_str),
                                        None,
                                    )

                                    all_behaviors[net_id].append(bi)

    for yamlfilename, behaviorname, overlaylabel in args.heuristic_behavior:
        with open(yamlfilename, 'r') as heuristic_file:
            video_docs = list(yaml.safe_load_all(heuristic_file))
        
        for video_doc in video_docs:
            net_id = video_doc['network_filename']
            if net_id in net_ids and behaviorname in video_doc:
                for bi_dict in video_doc[behaviorname]:
                    bi = BehaviorInterval(
                        bi_dict['start_frame'],
                        bi_dict['stop_frame_exclu'],
                        overlaylabel,
                        all_track_to_id_mapping[net_id][bi_dict['track1_id']],
                        all_track_to_id_mapping[net_id][bi_dict['track2_id']],
                    )

                    # print('===', overlaylabel, bi_dict['start_frame'], '===')

                    all_behaviors[net_id].append(bi)

    for behavior_intervals in all_behaviors.values():
        behavior_intervals.sort(key=lambda bi: bi.start_frame)

    for net_id in net_ids:
        print('processing:', net_id)
        net_id_root, _ = os.path.splitext(net_id)

        vid_intervals = all_behaviors[net_id]
        interval_cursor = 0
        active_intervals = []
        vid_path = os.path.join(args.video_root_dir, net_id)
        out_video_path = os.path.join(args.out_dir, net_id)
        os.makedirs(
            os.path.dirname(out_video_path),
            exist_ok=True,
        )

        pose_path = os.path.join(args.video_root_dir, net_id_root + '_pose_est_v3.h5')
        with h5py.File(pose_path, 'r') as pose_data:
            all_points = pose_data['poseest/points'][:]
            all_instance_count = pose_data['poseest/instance_count'][:]
            all_instance_track_id = pose_data['poseest/instance_track_id'][:]
            all_points_mask = pose_data['poseest/confidence'][:] > 0

        with imageio.get_reader(vid_path) as video_reader, \
                imageio.get_writer(out_video_path, fps=30) as video_writer:
            
            for frame_index, frame in enumerate(video_reader):

                # update the active intervals
                for i in range(interval_cursor, len(vid_intervals)):
                    curr_interval = vid_intervals[i]
                    if frame_index >= curr_interval.start_frame:
                        active_intervals.append(vid_intervals[i])
                        interval_cursor += 1
                    else:
                        break

                for i in reversed(range(len(active_intervals))):
                    curr_interval = active_intervals[i]
                    if frame_index > curr_interval.stop_frame_exclu:
                        del active_intervals[i]

                # render the overlay on this frame and save it to output video
                id_to_labels = dict()
                for bi in active_intervals:
                    if bi.track1_id not in id_to_labels:
                        id_to_labels[bi.track1_id] = []
                    id_to_labels[bi.track1_id].append(bi.behavior_label)

                    if bi.track2_id is not None:
                        if bi.track2_id not in id_to_labels:
                            id_to_labels[bi.track2_id] = []
                        id_to_labels[bi.track2_id].append(bi.behavior_label + ' (object)')

                for labels in id_to_labels.values():
                    labels.sort()

                curr_instance_count = all_instance_count[frame_index]
                for instance_index in range(curr_instance_count):
                    tracklet_id = all_instance_track_id[frame_index, instance_index]
                    mouse_id = all_track_to_id_mapping[net_id][tracklet_id]

                    # if mouse_id in id_to_labels:
                    #     print(id_to_labels[mouse_id])
                    if mouse_id in id_to_labels and all_points_mask[frame_index, instance_index, CENTER_SPINE_INDEX]:
                        center_spine_y, center_spine_x = all_points[frame_index, instance_index, CENTER_SPINE_INDEX, :]
                        cv2.circle(
                            frame,
                            (center_spine_x, center_spine_y),
                            5,
                            OVERLAY_COLOR,
                            cv2.FILLED,
                        )
                        cv2.putText(
                            frame,
                            ', '.join(id_to_labels[mouse_id]),
                            (center_spine_x + 6, center_spine_y + 6),
                            cv2.FONT_HERSHEY_COMPLEX,
                            1.0,
                            OVERLAY_COLOR,
                        )

                video_writer.append_data(frame)


if __name__ == '__main__':
    main()
