import argparse
import csv
import h5py
import imageio
import multiprocessing as mp
import numpy as np
import os
import yaml

import socialutil


def gen_instance_tracks(data_file_name, social_config):

    # extract some config stuff
    pose_config = social_config['pose']

    fps = pose_config['frames_per_sec']
    min_duration_secs = pose_config['min_track_duration_sec']
    min_duration_frames = min_duration_secs * fps

    pixels_per_cm = pose_config['pixels_per_cm']

    maximum_point_distance_cm = pose_config['interpolation']['maximum_point_distance_cm']
    maximum_point_distance_px = maximum_point_distance_cm * pixels_per_cm

    maximum_mean_point_distance_cm = pose_config['interpolation']['maximum_mean_point_distance_cm']
    maximum_mean_point_distance_px = maximum_mean_point_distance_cm * pixels_per_cm

    with h5py.File(data_file_name, 'r') as pose_h5:
        # extract data from the HDF5 file
        vid_grp = pose_h5['poseest']
        major_version = vid_grp.attrs['version'][0]

        assert major_version == 3

        all_points = vid_grp['points'][:]
        all_confidence = vid_grp['confidence'][:]
        all_instance_count = vid_grp['instance_count'][:]
        all_track_id = vid_grp['instance_track_id'][:]

        # build instance tracks from the HDF5 matrixes and
        track_dict = socialutil.build_tracks(
            all_points, all_confidence, all_instance_count, all_track_id)
        track_dict = {
            t_id: t for t_id, t
            in track_dict.items() if len(t['points']) >= min_duration_frames}

        # some post processing (interpolation etc)
        for track in track_dict.values():
            socialutil.interpolate_missing_points(
                track, maximum_point_distance_px, maximum_mean_point_distance_px)
            socialutil.calculate_point_velocities(track)
            socialutil.calculate_convex_hulls(track)

        return track_dict


def detect_chase_events(track_relationships, social_config):
    chase_config = social_config['behavior']['chase']

    pose_config = social_config['pose']
    fps = pose_config['frames_per_sec']
    pixels_per_cm = pose_config['pixels_per_cm']

    maximum_distance_px = chase_config['maximum_distance_cm'] * pixels_per_cm
    minimum_speed_px_frame = chase_config['minimum_speed_cm_sec'] * pixels_per_cm / fps
    minimum_duration_frames = chase_config['minimum_duration_sec'] * fps
    maximum_gap_frames = chase_config['maximum_gap_sec'] * fps
    maximum_chase_agreement = chase_config['minimum_chase_direction_agreement']

    for track_relationship in track_relationships:
        chase = socialutil.detect_chase(
            track_relationship,
            maximum_distance_px,
            minimum_speed_px_frame,
            minimum_duration_frames,
            maximum_gap_frames)
        chase_intervals = socialutil.find_intervals(chase, True)

        chase_intervals = socialutil.merge_intervals(chase_intervals, maximum_gap_frames)
        chase_intervals = [
            (start, stop)
            for start, stop in chase_intervals
            if stop - start >= minimum_duration_frames
        ]
        if chase_intervals:
            print('----')
            print(chase_intervals, track_relationship['stop_frame_exclu'] - track_relationship['start_frame'])
            for chase_interval in chase_intervals:

                # The chase interval is relative to the track_relationship start frame.
                # Let's change it so that that it's absolute frame index
                chase_start, chase_stop = chase_interval
                chase_start += track_relationship['start_frame']
                chase_stop += track_relationship['start_frame']

                # figure out who is chasing and who is being chased
                chaser_track_id = -1
                chasee_track_id = -1
                chase_direction = socialutil.detect_chase_direction(
                    track_relationship, chase_interval).mean()
                if chase_direction >= 0.5:
                    if chase_direction >= maximum_chase_agreement:
                        chaser_track_id = track_relationship['track1']['track_id']
                        chasee_track_id = track_relationship['track2']['track_id']
                else:
                    if 1 - chase_direction >= maximum_chase_agreement:
                        chaser_track_id = track_relationship['track2']['track_id']
                        chasee_track_id = track_relationship['track1']['track_id']

                if chaser_track_id != -1:
                    yield {
                        'chaser_track_id': chaser_track_id,
                        'chasee_track_id': chasee_track_id,
                        'chase_start_frame': chase_start,
                        'chase_stop_frame_exclu': chase_stop,
                    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--social-config',
        help='YAML file for configuring social behavior parameters',
        required=True,
    )
    parser.add_argument(
        '--batch-file',
        help='the batch file to process.',
        required=True,
    )
    parser.add_argument(
        '--root-dir',
        help='the root directory for the batch file',
        default='.',
    )
    parser.add_argument(
        '--out-file',
        help='the output HDF5 file to use',
        required=True,
    )
    args = parser.parse_args()

    with open(args.social_config) as social_config_file:
        social_config = yaml.safe_load(social_config_file)

    data_file_names = []
    with open(args.batch_file, newline='') as batch_file:
        batch_reader = csv.reader(batch_file, delimiter='\t')
        for row in batch_reader:
            if row:
                net_file_name = row[0]
                data_file_base, _ = os.path.splitext(net_file_name)
                data_file_name = os.path.join(args.root_dir, data_file_base + '_pose_est_v3.h5')
                data_file_names.append((net_file_name, data_file_name))

    outdir = os.path.dirname(args.out_file)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    with h5py.File(args.out_file, 'w') as social_h5:
        for net_file_name, data_file_name in data_file_names:
            print(net_file_name)

            instance_tracks = gen_instance_tracks(data_file_name, social_config)
            track_relationships = list(socialutil.calc_track_relationships(
                sorted(instance_tracks.values(), key=lambda track: track['start_frame'])))
            all_chases = list(detect_chase_events(track_relationships, social_config))


if __name__ == "__main__":
    main()
