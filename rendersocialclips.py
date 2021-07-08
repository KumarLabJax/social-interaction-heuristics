import argparse
from collections import deque
import cv2
import h5py
import imageio
import multiprocessing as mp
import numpy as np
import os
import urllib.parse as urlparse
import yaml

import gensocialstats
import rendervidoverlay
import socialutil


CHASER_NON_CHASE_COLOR = (255 // 2, 0, 0)
CHASER_CHASE_COLOR = (255, 0, 0)

CHASEE_NON_CHASE_COLOR = (0, 255 // 2, 0)
CHASEE_CHASE_COLOR = (0, 255, 0)

# color palette from https://colorbrewer2.org/?type=qualitative&scheme=Accent&n=7
TRACK_INSTANCE_COLORS = [
    # (127,201,127),
    (190,174,212),
    (253,192,134),
    (255,255,153),
    (56,108,176),
    # (240,2,127),
    (191,91,23),
]


BEHAVIOR_NAMES = [
    # 'oral_genital_contact',
    # 'oral_oral_contact',
    # 'oral_ear_contact',
    # 'chases',
    # 'approaches',
    # 'huddles',

    'watching',
    # 'close',
    # 'contact',
]


BEHAVIOR_ANNOTATION_INFO = {
    # 'oral_genital_contact': {
    #     'horizontal_offset_px': 0,
    #     'vertical_offset_px': 30,
    #     'TEXT': 'OG',
    # },
    # 'oral_oral_contact': {
    #     'horizontal_offset_px': 300,
    #     'vertical_offset_px': 30,
    #     'TEXT': 'OO',
    # },
    # 'oral_ear_contact': {
    #     'horizontal_offset_px': 600,
    #     'vertical_offset_px': 30,
    #     'TEXT': 'OE',
    # },
    # 'chases': {
    #     'horizontal_offset_px': 0,
    #     'vertical_offset_px': 60,
    #     'TEXT': 'CH',
    # },
    # 'approaches': {
    #     'horizontal_offset_px': 300,
    #     'vertical_offset_px': 60,
    #     'TEXT': 'AP',
    # },
    # 'huddles': {
    #     'horizontal_offset_px': 600,
    #     'vertical_offset_px': 60,
    #     'TEXT': 'HU',
    # },

    'watching': {
        'horizontal_offset_px': 0,
        'vertical_offset_px': 30,
        'TEXT': 'WA',
    },
    # 'close': {
    #     'horizontal_offset_px': 300,
    #     'vertical_offset_px': 30,
    #     'TEXT': 'CL',
    # },
    # 'contact': {
    #     'horizontal_offset_px': 600,
    #     'vertical_offset_px': 30,
    #     'TEXT': 'CO',
    # },
}


VIDEO_ANNOTATION_PADDING_PX = 112

TEXT_HEIGHT_PX = 22
TEXT_WIDTH_PX = 44

FRAME_NUM_VERTICAL_OFFSET_PX = 90

def render_overlay(frame, pose, pose_mask, exclude_points, color):

    zero_conf_indexes = set((~pose_mask).nonzero()[0])
    inst_exclude_points = exclude_points | zero_conf_indexes
    rendervidoverlay.render_pose_overlay(frame, pose, inst_exclude_points, color)


def frame_pose(track, frame_index):
    track_start_frame = track['start_frame']
    track_stop_frame_exclu = track['stop_frame_exclu']
    track_pose = None
    track_pose_mask = None
    if track_start_frame <= frame_index < track_stop_frame_exclu:
        pose_index = frame_index - track_start_frame
        track_pose = track['points'][pose_index, ...]
        track_pose_mask = track['point_masks'][pose_index, ...]

    return track_pose, track_pose_mask


class InteractionVideoClip(object):

    def __init__(
            self,
            behavior_name,
            track1,
            track2,
            interaction_start_frame,
            interaction_stop_frame_exclu,
            exclude_points,
            buffer_frames=30):

        self.behavior_name = behavior_name

        self.track1 = track1
        self.track2 = track2
        self.interaction_start_frame = interaction_start_frame
        self.interaction_stop_frame_exclu = interaction_stop_frame_exclu
        self.buffer_frames = buffer_frames

        self.exclude_points = exclude_points

    @property
    def start_frame(self):
        return max(self.interaction_start_frame - self.buffer_frames, 0)

    @property
    def stop_frame_exclu(self):
        return self.interaction_stop_frame_exclu + self.buffer_frames

    # def process_frame(self, behavior_name, frame, curr_frame_index):

    #     if self.behavior_name is None or self.behavior_name == behavior_name:

    #         if self.start_frame <= curr_frame_index < self.stop_frame_exclu:

    #             if self.behavior_name is not None:
    #                 interaction_active = (
    #                     self.interaction_start_frame <= curr_frame_index < self.interaction_stop_frame_exclu
    #                 )
    #                 pose1, pose_mask1 = frame_pose(self.track1, curr_frame_index)
    #                 if pose1 is not None:
    #                     render_overlay(
    #                         frame,
    #                         pose1, pose_mask1,
    #                         self.exclude_points,
    #                         CHASER_CHASE_COLOR if interaction_active else CHASER_NON_CHASE_COLOR)

    #                 pose2, pose_mask2 = frame_pose(self.track2, curr_frame_index)
    #                 if pose2 is not None:
    #                     render_overlay(
    #                         frame,
    #                         pose2, pose_mask2,
    #                         self.exclude_points,
    #                         CHASEE_CHASE_COLOR if interaction_active else CHASEE_NON_CHASE_COLOR)

    #             return True

    #         else:
    #             return False

    #     else:
    #         return False


# share_root=/media/sheppk/TOSHIBA\ EXT/cached-data/BTBR_3M_stranger_4day
# python -u rendersocialclips.py \
#       --social-config social-config.yaml \
#       --social-file-in BTBR_3M_stranger_4day-social-2020-04-22.yaml \
#       --root-dir "${share_root}" \
#       --out-dir tempout3 \
#       --allow-missing-video

# share_root=/home/sheppk/smb/labshare
# python -u rendersocialclips.py \
#       --social-config social-config.yaml \
#       --social-file-in UCSD-out-2020-06-15.yaml \
#       --root-dir "${share_root}" \
#       --out-dir UCSD-out-2020-06-15-clips \
#       --allow-missing-video

# share_root=/home/sheppk/smb/labshare
# python -u rendersocialclips.py \
#       --social-config social-config.yaml \
#       --social-file-in UCSD-out-2020-08-04.yaml \
#       --root-dir "${share_root}" \
#       --out-dir UCSD-out-2020-08-04-clips \
#       --allow-missing-video \
#       --proximity-threshold-cm 3

# share_root=/home/sheppk/smb/labshare
# python -u rendersocialclips.py \
#       --social-config social-config.yaml \
#       --social-file-in UCSD-out-2020-08-04.yaml \
#       --root-dir "${share_root}" \
#       --out-dir UCSD-out-2020-08-24-clips \
#       --allow-missing-video \
#       --proximity-threshold-cm 3

# share_root=/home/sheppk/smb/labshare
# python -u rendersocialclips.py \
#       --social-config social-config.yaml \
#       --social-file-in UCSD-out-2020-08-04.yaml \
#       --root-dir "${share_root}" \
#       --out-dir UCSD-out-2020-08-24-clips \
#       --allow-missing-video \
#       --proximity-threshold-cm 3 \
#       --print-frame-count

# python -u rendersocialclips.py \
#       --social-config social-config.yaml \
#       --social-file-in <(cat data/*2021-06-11-social.yaml) \
#       --root-dir "/media/sheppk/TOSHIBA EXT/rotta-data/B6J-and-BTBR-3M-strangers-4-day-rand-2021-05-24" \
#       --out-dir "/media/sheppk/TOSHIBA EXT/rotta-data/B6J-and-BTBR-3M-strangers-4-day-rand-2021-05-24-watching-close-contact" \
#       --allow-missing-video \
#       --proximity-threshold-cm 3

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--social-config',
        help='YAML file for configuring social behavior parameters',
        required=True,
    )

    parser.add_argument(
        '--social-file-in',
        help='the YAML file with social behavior inference',
        required=True,
    )

    parser.add_argument(
        '--root-dir',
        help='input root directory for videos and pose files',
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
    parser.add_argument(
        '--proximity-threshold-cm',
        help='An interval is always rendered when proximity is under this threshold'
             ' (in cm) even in cases where there is no social interaction.',
        type=float,
        default=0,
    )

    args = parser.parse_args()

    with open(args.social_config) as social_config_file:
        social_config = yaml.safe_load(social_config_file)

    pose_config = social_config['pose']
    pixels_per_cm = pose_config['pixels_per_cm']
    proximity_thresh_px = args.proximity_threshold_cm * pixels_per_cm

    exclude_points = set()
    exclude_points.add(rendervidoverlay.LEFT_FRONT_PAW_INDEX)
    exclude_points.add(rendervidoverlay.RIGHT_FRONT_PAW_INDEX)
    exclude_points.add(rendervidoverlay.LEFT_EAR_INDEX)
    exclude_points.add(rendervidoverlay.RIGHT_EAR_INDEX)

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.social_file_in, 'r') as social_file:
        for video_doc in yaml.safe_load_all(social_file):
            net_id = video_doc['network_filename']
            escaped_net_id = urlparse.quote(net_id, safe='')
            escaped_net_id_root, _ = os.path.splitext(escaped_net_id)
            in_video_path = os.path.join(args.root_dir, net_id)

            if args.allow_missing_video:
                if not os.path.exists(in_video_path):
                    print('WARNING: ' + in_video_path + ' does not exist')
                    continue
            else:
                assert os.path.exists(in_video_path), in_video_path + ' does not exist'

            print('PROCESSING:', net_id)

            file_no_ext, _ = os.path.splitext(in_video_path)
            pose_file_name = file_no_ext + '_pose_est_v4.h5'
            if not os.path.exists(pose_file_name):
                pose_file_name = file_no_ext + '_pose_est_v3.h5'
            assert os.path.exists(pose_file_name)

            behavior_intervals = []

            tracks, frame_count = gensocialstats.gen_instance_tracks(pose_file_name, social_config)
            sorted_track_list = sorted(tracks.values(), key=lambda track: track['start_frame'])

            # allocate colors to tracks such that the same color is never used in the same time frame
            available_color_queue = deque(TRACK_INSTANCE_COLORS)
            track_color_dict = dict()
            active_tracks = []
            for curr_track in sorted_track_list:
                curr_start_frame = curr_track['start_frame']
                for i in reversed(range(len(active_tracks))):
                    if active_tracks[i]['stop_frame_exclu'] <= curr_start_frame:

                        # we have moved passed this active frame so we can make its
                        # track color available and delete it from the active list
                        available_color_queue.append(track_color_dict[active_tracks[i]['track_id']])
                        del active_tracks[i]

                if available_color_queue:
                    track_color_dict[curr_track['track_id']] = available_color_queue.popleft()
                    active_tracks.append(curr_track)
                else:
                    # default to black if we've run out of colors
                    track_color_dict[curr_track['track_id']] = (0, 0, 0)

            if proximity_thresh_px > 0:
                track_relationships = list(socialutil.calc_track_relationships(sorted_track_list))

                proximity_thresh_arr = np.zeros(video_doc['frame_count'], dtype=np.bool)
                for tr in track_relationships:
                    start = tr['start_frame']
                    stop = tr['stop_frame_exclu']
                    proximity_thresh_arr[start:stop] = np.logical_or(
                        proximity_thresh_arr[start:stop],
                        tr['track_distances'] <= proximity_thresh_px,
                    )

                proximity_intervals = socialutil.find_intervals(proximity_thresh_arr, True)
                proximity_intervals = socialutil.merge_intervals(proximity_intervals, 30 * 5)
                for pi_start, pi_stop in proximity_intervals:
                    behavior_intervals.append(InteractionVideoClip(
                        None,
                        None,
                        None,
                        pi_start,
                        pi_stop,
                        exclude_points,
                    ))

            if 'oral_genital_contact' in BEHAVIOR_ANNOTATION_INFO:
                for og_contact_index, og_contact in enumerate(video_doc['oral_genital_contact']):
                    track1 = tracks[og_contact['track1_id']]
                    track2 = tracks[og_contact['track2_id']]

                    curr_contact = InteractionVideoClip(
                        'oral_genital_contact',
                        track1,
                        track2,
                        og_contact['start_frame'],
                        og_contact['stop_frame_exclu'],
                        exclude_points,
                    )
                    behavior_intervals.append(curr_contact)

            if 'oral_oral_contact' in BEHAVIOR_ANNOTATION_INFO:
                for oo_contact_index, oo_contact in enumerate(video_doc['oral_oral_contact']):
                    track1 = tracks[oo_contact['track1_id']]
                    track2 = tracks[oo_contact['track2_id']]

                    curr_contact = InteractionVideoClip(
                        'oral_oral_contact',
                        track1,
                        track2,
                        oo_contact['start_frame'],
                        oo_contact['stop_frame_exclu'],
                        exclude_points,
                    )
                    behavior_intervals.append(curr_contact)

            if 'oral_ear_contact' in BEHAVIOR_ANNOTATION_INFO:
                for oe_contact_index, oe_contact in enumerate(video_doc['oral_ear_contact']):
                    track1 = tracks[oe_contact['track1_id']]
                    track2 = tracks[oe_contact['track2_id']]

                    curr_contact = InteractionVideoClip(
                        'oral_ear_contact',
                        track1,
                        track2,
                        oe_contact['start_frame'],
                        oe_contact['stop_frame_exclu'],
                        exclude_points,
                    )
                    behavior_intervals.append(curr_contact)

            if 'chases' in BEHAVIOR_ANNOTATION_INFO:
                for chase_index, chase in enumerate(video_doc['chases']):
                    chaser_track = tracks[chase['chaser_track_id']]
                    chasee_track = tracks[chase['chasee_track_id']]

                    curr_chase = InteractionVideoClip(
                        'chases',
                        chaser_track,
                        chasee_track,
                        chase['start_frame'],
                        chase['stop_frame_exclu'],
                        exclude_points,
                    )
                    behavior_intervals.append(curr_chase)

            if 'approaches' in BEHAVIOR_ANNOTATION_INFO:
                for approach_index, approach in enumerate(video_doc['approaches']):
                    approacher_track = tracks[approach['approacher_track_id']]
                    approached_track = tracks[approach['approached_track_id']]

                    curr_approach = InteractionVideoClip(
                        'approaches',
                        approacher_track,
                        approached_track,
                        approach['start_frame'],
                        approach['stop_frame_exclu'],
                        exclude_points,
                    )
                    behavior_intervals.append(curr_approach)

            if 'huddles' in BEHAVIOR_ANNOTATION_INFO:
                for huddle_index, huddle in enumerate(video_doc['huddles']):
                    track1 = tracks[huddle['track1_id']]
                    track2 = tracks[huddle['track2_id']]

                    curr_huddle = InteractionVideoClip(
                        'huddles',
                        track1,
                        track2,
                        huddle['start_frame'],
                        huddle['stop_frame_exclu'],
                        exclude_points,
                    )
                    behavior_intervals.append(curr_huddle)

            if 'watching' in BEHAVIOR_ANNOTATION_INFO:
                for watching in video_doc['watching']:
                    track1 = tracks[watching['subject_track_id']]
                    track2 = tracks[watching['object_track_id']]

                    curr_watching = InteractionVideoClip(
                        'watching',
                        track1,
                        track2,
                        watching['start_frame'],
                        watching['stop_frame_exclu'],
                        exclude_points,
                    )
                    behavior_intervals.append(curr_watching)

            if 'close' in BEHAVIOR_ANNOTATION_INFO:
                for close_interval in video_doc['close']:
                    track1 = tracks[close_interval['track1_id']]
                    track2 = tracks[close_interval['track2_id']]

                    curr_close = InteractionVideoClip(
                        'close',
                        track1,
                        track2,
                        close_interval['start_frame'],
                        close_interval['stop_frame_exclu'],
                        exclude_points,
                    )
                    behavior_intervals.append(curr_close)

            if 'contact' in BEHAVIOR_ANNOTATION_INFO:
                for contact_interval in video_doc['contact']:
                    track1 = tracks[contact_interval['track1_id']]
                    track2 = tracks[contact_interval['track2_id']]

                    curr_contact = InteractionVideoClip(
                        'contact',
                        track1,
                        track2,
                        contact_interval['start_frame'],
                        contact_interval['stop_frame_exclu'],
                        exclude_points,
                    )
                    behavior_intervals.append(curr_contact)

            if behavior_intervals:
                out_video_path = os.path.join(
                        args.out_dir,
                        escaped_net_id_root + '_social.avi',
                    )

                with imageio.get_reader(in_video_path) as video_reader, \
                     imageio.get_writer(out_video_path, fps=30) as video_writer:
                    for frame_index, frame in enumerate(video_reader):
                        frame_copy = frame.copy()
                        for track in sorted_track_list:
                            pose, pose_mask = frame_pose(track, frame_index)
                            if pose is not None:
                                render_overlay(
                                    frame_copy,
                                    pose, pose_mask,
                                    exclude_points,
                                    track_color_dict[track['track_id']])

                        frame_row_count, frame_col_count, frame_color_count = frame_copy.shape
                        annotation_padding = np.zeros((VIDEO_ANNOTATION_PADDING_PX, frame_col_count, frame_color_count), dtype=np.uint8)
                        frame_copy = np.append(frame_copy, annotation_padding, axis=0)

                        write_frame = annotate_frame(frame_copy, frame_index, behavior_intervals)
                        if write_frame:
                            video_writer.append_data(frame_copy)


def annotate_frame(frame, frame_index, behavior_intervals):
    render_frame = False

    frame_row_count, frame_col_count, frame_color_count = frame.shape

    annotation_start_row = frame_row_count - VIDEO_ANNOTATION_PADDING_PX

    for bi in behavior_intervals:
        if bi.start_frame <= frame_index < bi.stop_frame_exclu:

            if bi.behavior_name is not None and bi.behavior_name in BEHAVIOR_ANNOTATION_INFO:
                interaction_active = (
                    bi.interaction_start_frame <= frame_index < bi.interaction_stop_frame_exclu
                )

                anno_info = BEHAVIOR_ANNOTATION_INFO[bi.behavior_name]
                bn_x = anno_info['horizontal_offset_px']
                bn_y = annotation_start_row + anno_info['vertical_offset_px']
                cv2.rectangle(
                    frame,
                    (bn_x + TEXT_WIDTH_PX, bn_y - TEXT_HEIGHT_PX),
                    (bn_x + TEXT_WIDTH_PX + TEXT_HEIGHT_PX, bn_y),
                    (255, 255, 255),
                    cv2.FILLED,
                )

            render_frame = True

    if render_frame:
        cv2.putText(
            frame,
            'Frame #: {}'.format(frame_index + 1),
            (0, annotation_start_row + FRAME_NUM_VERTICAL_OFFSET_PX),
            cv2.FONT_HERSHEY_COMPLEX,
            1.0,
            (255, 255, 255),
        )

        for bn in BEHAVIOR_NAMES:
            anno_info = BEHAVIOR_ANNOTATION_INFO[bn]
            bn_x = anno_info['horizontal_offset_px']
            bn_y = annotation_start_row + anno_info['vertical_offset_px']
            cv2.putText(
                frame,
                anno_info['TEXT'],
                (bn_x, bn_y),
                cv2.FONT_HERSHEY_COMPLEX,
                1.0,
                (255, 255, 255),
            )
            cv2.rectangle(
                frame,
                (bn_x + TEXT_WIDTH_PX, bn_y - TEXT_HEIGHT_PX),
                (bn_x + TEXT_WIDTH_PX + TEXT_HEIGHT_PX, bn_y),
                (255, 255, 255),
                1,
            )

    return render_frame


if __name__ == '__main__':
    main()
