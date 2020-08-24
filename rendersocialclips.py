import argparse
from collections import deque
import h5py
import imageio
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

# # these colors
# TRACK_INSTANCE_COLORS = [
#     (141, 211, 199),
#     (255, 255, 179),
#     (190, 186, 218),
#     (251, 128, 114),
#     (128, 177, 211),
#     (253, 180, 98),
#     (179, 222, 105),
#     (252, 205, 229),
# ]

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
    'oral_genital_contact',
    'oral_oral_contact',
    'chases',
    'approaches',
    'huddles',
]


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

    def process_frame(self, behavior_name, frame, curr_frame_index):

        if self.behavior_name is None or self.behavior_name == behavior_name:

            if self.start_frame <= curr_frame_index < self.stop_frame_exclu:

                if self.behavior_name is not None:
                    interaction_active = (
                        self.interaction_start_frame <= curr_frame_index < self.interaction_stop_frame_exclu
                    )
                    pose1, pose_mask1 = frame_pose(self.track1, curr_frame_index)
                    if pose1 is not None:
                        render_overlay(
                            frame,
                            pose1, pose_mask1,
                            self.exclude_points,
                            CHASER_CHASE_COLOR if interaction_active else CHASER_NON_CHASE_COLOR)

                    pose2, pose_mask2 = frame_pose(self.track2, curr_frame_index)
                    if pose2 is not None:
                        render_overlay(
                            frame,
                            pose2, pose_mask2,
                            self.exclude_points,
                            CHASEE_CHASE_COLOR if interaction_active else CHASEE_NON_CHASE_COLOR)

                return True

            else:
                return False

        else:
            return False


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
                    # print('WARNING: ' + in_video_path + ' does not exist')
                    continue
            else:
                assert os.path.exists(in_video_path), in_video_path + ' does not exist'

            print('PROCESSING:', net_id)

            file_no_ext, _ = os.path.splitext(in_video_path)
            pose_file_name = file_no_ext + '_pose_est_v3.h5'
            assert os.path.exists(pose_file_name)

            interaction_clips = []

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
                    interaction_clips.append(InteractionVideoClip(
                        None,
                        None,
                        None,
                        pi_start,
                        pi_stop,
                        exclude_points,
                    ))

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
                interaction_clips.append(curr_contact)

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
                interaction_clips.append(curr_contact)

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
                interaction_clips.append(curr_chase)

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
                interaction_clips.append(curr_approach)

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
                interaction_clips.append(curr_huddle)

            if interaction_clips:
                def vid_out_path(behavior_name):
                    return os.path.join(
                        args.out_dir,
                        escaped_net_id_root + '_' + behavior_name + '.avi',
                    )
                vid_writers = {
                    behavior_name: imageio.get_writer(vid_out_path(behavior_name), fps=30)
                    for behavior_name in BEHAVIOR_NAMES
                }

                with imageio.get_reader(in_video_path) as video_reader:
                    # active_clips = []
                    for frame_num, frame in enumerate(video_reader):
                        for bn in BEHAVIOR_NAMES:
                            frame_copy = frame.copy()

                            for track in sorted_track_list:
                                pose, pose_mask = frame_pose(track, frame_num)
                                if pose is not None:
                                    render_overlay(
                                        frame_copy,
                                        pose, pose_mask,
                                        exclude_points,
                                        track_color_dict[track['track_id']])

                            any_interaction = False
                            for ic in interaction_clips:
                                interaction_valid = ic.process_frame(bn, frame_copy, frame_num)
                                if interaction_valid:
                                    any_interaction = True

                            if any_interaction:
                                vid_writers[bn].append_data(frame_copy)

                for writer in vid_writers.values():
                    writer.close()


if __name__ == '__main__':
    main()
