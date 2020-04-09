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


class ChaseVideoClip(object):

    def __init__(
            self,
            out_file_name,
            chaser_track,
            chasee_track,
            chase_start_frame,
            chase_stop_frame_exclu,
            exclude_points,
            buffer_frames=30):

        self.out_file_name = out_file_name
        self.out_file_writer = None

        self.chaser_track = chaser_track
        self.chasee_track = chasee_track
        self.chase_start_frame = chase_start_frame
        self.chase_stop_frame_exclu = chase_stop_frame_exclu
        self.buffer_frames = buffer_frames

        self.exclude_points = exclude_points

        self.curr_frame_index = self.start_frame

    @property
    def start_frame(self):
        return max(self.chase_start_frame - self.buffer_frames, 0)

    @property
    def stop_frame_exclu(self):
        return self.chase_stop_frame_exclu + self.buffer_frames

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

        chase_active = (
            self.chase_start_frame <= self.curr_frame_index < self.chase_stop_frame_exclu
        )
        chaser_pose, chaser_pose_mask = self._curr_frame_pose(self.chaser_track)
        if chaser_pose is not None:
            self.render_overlay(
                frame,
                chaser_pose, chaser_pose_mask,
                CHASER_CHASE_COLOR if chase_active else CHASER_NON_CHASE_COLOR)

        chasee_pose, chasee_pose_mask = self._curr_frame_pose(self.chasee_track)
        if chasee_pose is not None:
            self.render_overlay(
                frame,
                chasee_pose, chasee_pose_mask,
                CHASEE_CHASE_COLOR if chase_active else CHASEE_NON_CHASE_COLOR)

        print('writing frame', self.curr_frame_index)
        self.out_file_writer.append_data(frame)

        self.curr_frame_index += 1

    def close(self):
        if self.out_file_writer is not None:
            self.out_file_writer.close()
            self.out_file_writer = None


# share_root="/run/user/1002/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar"
# python -u rendersocialclips.py \
#   --social-config social-config.yaml \
#   --social-file-in tempout.h5 \
#   --root-dir "${share_root}/VideoData/MDS_Tests/BTBR_3M_stranger_4day"

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--social-config',
        help='YAML file for configuring social behavior parameters',
        required=True,
    )
    # parser.add_argument(
    #     '--exclude-forepaws',
    #     action='store_true',
    #     dest='exclude_forepaws',
    #     default=False,
    #     help='should we exclude the forepaws',
    # )
    # parser.add_argument(
    #     '--exclude-ears',
    #     action='store_true',
    #     dest='exclude_ears',
    #     default=False,
    #     help='should we exclude the ears',
    # )

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

    args = parser.parse_args()

    with open(args.social_config) as social_config_file:
        social_config = yaml.safe_load(social_config_file)

    # exclude_points = set()
    # if args.exclude_forepaws:
    #     exclude_points.add(rendervidoverlay.LEFT_FRONT_PAW_INDEX)
    #     exclude_points.add(rendervidoverlay.RIGHT_FRONT_PAW_INDEX)
    # if args.exclude_ears:
    #     exclude_points.add(rendervidoverlay.LEFT_EAR_INDEX)
    #     exclude_points.add(rendervidoverlay.RIGHT_EAR_INDEX)
    exclude_points = set()
    exclude_points.add(rendervidoverlay.LEFT_FRONT_PAW_INDEX)
    exclude_points.add(rendervidoverlay.RIGHT_FRONT_PAW_INDEX)
    exclude_points.add(rendervidoverlay.LEFT_EAR_INDEX)
    exclude_points.add(rendervidoverlay.RIGHT_EAR_INDEX)

    with h5py.File(args.social_file_in, 'r') as social_file:
        for escaped_net_id, social_grp in social_file.items():
            net_id = urlparse.unquote(escaped_net_id)
            net_id_no_ext, _ = os.path.splitext(net_id)
            if len(social_grp['chase_frame_count']) > 0:
                print(net_id)
                print(social_grp['chase_track_ids'][:].shape)
                print(social_grp['chase_start_frame'][:].shape)
                print(social_grp['chase_frame_count'][:].shape)

                chase_track_ids = social_grp['chase_track_ids'][:]
                chase_start_frames = social_grp['chase_start_frame'][:]
                chase_frame_counts = social_grp['chase_frame_count'][:]

                in_video_path = os.path.join(args.root_dir, net_id)
                assert os.path.exists(in_video_path)

                file_no_ext, _ = os.path.splitext(in_video_path)
                pose_file_name = file_no_ext + '_pose_est_v3.h5'
                assert os.path.exists(pose_file_name)

                tracks = gensocialstats.gen_instance_tracks(pose_file_name, social_config)
                print(len(tracks))

                chase_vid_clips = dict()
                for chase_index in range(len(chase_start_frames)):
                    chaser_track_id, chasee_track_id = chase_track_ids[chase_index, :]
                    chaser_track = tracks[chaser_track_id]
                    chasee_track = tracks[chasee_track_id]

                    chase_start_frame = chase_start_frames[chase_index]
                    chase_frame_count = chase_frame_counts[chase_index]
                    # chase_start_frame = 2000
                    chase_stop_frame_exclu = chase_start_frame + chase_frame_count

                    out_file_name = os.path.join(
                        args.out_dir,
                        escaped_net_id + '_chase_' + str(chase_index) + '.avi',
                    )

                    curr_chase = ChaseVideoClip(
                        out_file_name,
                        chaser_track,
                        chasee_track,
                        chase_start_frame,
                        chase_stop_frame_exclu,
                        exclude_points,
                    )
                    chase_vid_clips[curr_chase.start_frame] = curr_chase

                # active_chase_vid_clips = []
                with imageio.get_reader(in_video_path) as video_reader:
                    active_clips = []
                    for frame_num, frame in enumerate(video_reader):
                        if frame_num in chase_vid_clips:
                            active_clips.append(chase_vid_clips[frame_num])

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
