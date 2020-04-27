import argparse
import h5py
import imageio
import numpy as np
import os
import urllib.parse as urlparse
import yaml

import gensocialstats
import rendervidoverlay


CHASER_NON_CHASE_COLOR = (255 // 2, 0, 0)
CHASER_CHASE_COLOR = (255, 0, 0)

CHASEE_NON_CHASE_COLOR = (0, 255 // 2, 0)
CHASEE_CHASE_COLOR = (0, 255, 0)


class InteractionVideoClip(object):

    def __init__(
            self,
            out_file_name,
            track1,
            track2,
            interaction_start_frame,
            interaction_stop_frame_exclu,
            exclude_points,
            buffer_frames=30):

        self.out_file_name = out_file_name
        self.out_file_writer = None

        self.track1 = track1
        self.track2 = track2
        self.interaction_start_frame = interaction_start_frame
        self.interaction_stop_frame_exclu = interaction_stop_frame_exclu
        self.buffer_frames = buffer_frames

        self.exclude_points = exclude_points

        self.curr_frame_index = self.start_frame

    @property
    def start_frame(self):
        return max(self.interaction_start_frame - self.buffer_frames, 0)

    @property
    def stop_frame_exclu(self):
        return self.interaction_stop_frame_exclu + self.buffer_frames

    def render_overlay(self, frame, pose, pose_mask, color):
        zero_conf_indexes = set((~pose_mask).nonzero()[0])
        inst_exclude_points = self.exclude_points | zero_conf_indexes
        rendervidoverlay.render_pose_overlay(frame, pose, inst_exclude_points, color)

    def _curr_frame_pose(self, track):
        track_start_frame = track['start_frame']
        track_stop_frame_exclu = track['stop_frame_exclu']
        track_pose = None
        track_pose_mask = None
        if track_start_frame <= self.curr_frame_index < track_stop_frame_exclu:
            pose_index = self.curr_frame_index - track_start_frame
            track_pose = track['points'][pose_index, ...]
            track_pose_mask = track['point_masks'][pose_index, ...]

        return track_pose, track_pose_mask

    def process_frame(self, frame):
        if self.out_file_writer is None:
            os.makedirs(os.path.dirname(self.out_file_name), exist_ok=True)
            self.out_file_writer = imageio.get_writer(self.out_file_name, fps=30)

        interaction_active = (
            self.interaction_start_frame <= self.curr_frame_index < self.interaction_stop_frame_exclu
        )
        pose1, pose_mask1 = self._curr_frame_pose(self.track1)
        if pose1 is not None:
            self.render_overlay(
                frame,
                pose1, pose_mask1,
                CHASER_CHASE_COLOR if interaction_active else CHASER_NON_CHASE_COLOR)

        pose2, pose_mask2 = self._curr_frame_pose(self.track2)
        if pose2 is not None:
            self.render_overlay(
                frame,
                pose2, pose_mask2,
                CHASEE_CHASE_COLOR if interaction_active else CHASEE_NON_CHASE_COLOR)

        print('writing frame', self.curr_frame_index)
        self.out_file_writer.append_data(frame)

        self.curr_frame_index += 1

    def close(self):
        if self.out_file_writer is not None:
            self.out_file_writer.close()
            self.out_file_writer = None


# share_root=/media/sheppk/TOSHIBA\ EXT/cached-data/BTBR_3M_stranger_4day
# python -u rendersocialclips.py \
#       --social-config social-config.yaml \
#       --social-file-in BTBR_3M_stranger_4day-social-2020-04-22.yaml \
#       --root-dir "${share_root}" \
#       --out-dir tempout3 \
#       --allow-missing-video
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--social-config',
        help='YAML file for configuring social behavior parameters',
        required=True,
    )

    parser.add_argument(
        '--social-file-in',
        help='the HDF5 file with social behavior inference',
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

    args = parser.parse_args()

    with open(args.social_config) as social_config_file:
        social_config = yaml.safe_load(social_config_file)

    exclude_points = set()
    exclude_points.add(rendervidoverlay.LEFT_FRONT_PAW_INDEX)
    exclude_points.add(rendervidoverlay.RIGHT_FRONT_PAW_INDEX)
    exclude_points.add(rendervidoverlay.LEFT_EAR_INDEX)
    exclude_points.add(rendervidoverlay.RIGHT_EAR_INDEX)

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
                    print('YO it exsts', in_video_path)
            else:
                assert os.path.exists(in_video_path), in_video_path + ' does not exist'

            file_no_ext, _ = os.path.splitext(in_video_path)
            pose_file_name = file_no_ext + '_pose_est_v3.h5'
            assert os.path.exists(pose_file_name)

            tracks = gensocialstats.gen_instance_tracks(pose_file_name, social_config)

            vid_clips = dict()
            for og_contact_index, og_contact in enumerate(video_doc['oral_genital_contact']):
                track1 = tracks[og_contact['track1_id']]
                track2 = tracks[og_contact['track2_id']]

                out_file_name = os.path.join(
                    args.out_dir,
                    escaped_net_id_root + '_og_contact_' + str(og_contact_index) +
                            '_' + str(og_contact['track1_id']) +
                            '_' + str(og_contact['track2_id']) + '.avi',
                )

                curr_contact = InteractionVideoClip(
                    out_file_name,
                    track1,
                    track2,
                    og_contact['start_frame'],
                    og_contact['stop_frame_exclu'],
                    exclude_points,
                )
                if curr_contact.start_frame in vid_clips:
                    vid_clips[curr_contact.start_frame].append(curr_contact)
                else:
                    vid_clips[curr_contact.start_frame] = [curr_contact]

            for oo_contact_index, oo_contact in enumerate(video_doc['oral_oral_contact']):
                track1 = tracks[oo_contact['track1_id']]
                track2 = tracks[oo_contact['track2_id']]

                out_file_name = os.path.join(
                    args.out_dir,
                    escaped_net_id_root + '_oo_contact_' + str(og_contact_index) +
                            '_' + str(oo_contact['track1_id']) +
                            '_' + str(oo_contact['track2_id']) + '.avi',
                )

                curr_contact = InteractionVideoClip(
                    out_file_name,
                    track1,
                    track2,
                    oo_contact['start_frame'],
                    oo_contact['stop_frame_exclu'],
                    exclude_points,
                )
                if curr_contact.start_frame in vid_clips:
                    vid_clips[curr_contact.start_frame].append(curr_contact)
                else:
                    vid_clips[curr_contact.start_frame] = [curr_contact]

            for chase_index, chase in enumerate(video_doc['chases']):
                chaser_track = tracks[chase['chaser_track_id']]
                chasee_track = tracks[chase['chasee_track_id']]

                out_file_name = os.path.join(
                    args.out_dir,
                    escaped_net_id_root + '_chase_' + str(chase_index) +
                            '_' + str(chase['chaser_track_id']) +
                            '_' + str(chase['chasee_track_id']) + '.avi',
                )

                curr_chase = InteractionVideoClip(
                    out_file_name,
                    chaser_track,
                    chasee_track,
                    chase['start_frame'],
                    chase['stop_frame_exclu'],
                    exclude_points,
                )
                if curr_chase.start_frame in vid_clips:
                    vid_clips[curr_chase.start_frame].append(curr_chase)
                else:
                    vid_clips[curr_chase.start_frame] = [curr_chase]

            if vid_clips:
                with imageio.get_reader(in_video_path) as video_reader:
                    active_clips = []
                    for frame_num, frame in enumerate(video_reader):
                        if frame_num in vid_clips:
                            active_clips.extend(vid_clips[frame_num])

                        deactive_clips = [c for c in active_clips if frame_num >= c.stop_frame_exclu]
                        active_clips = [c for c in active_clips if frame_num < c.stop_frame_exclu]
                        for c in deactive_clips:
                            c.close()

                        for c in active_clips:
                            c.process_frame(frame)

                    for c in active_clips:
                        c.close()


if __name__ == '__main__':
    main()
