import argparse
import csv
import functools
import h5py
import imageio
import itertools
import multiprocessing as mp
import numpy as np
import os
import urllib.parse as urlparse
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
    maximum_chase_norm_of_deviation = socialutil.norm_of_deviation(
        chase_config['maximum_chase_direction_deviation_deg'])

    for track_relationship in track_relationships:
        chase = socialutil.detect_chase(
            track_relationship,
            maximum_distance_px,
            minimum_speed_px_frame)
        chase_intervals = socialutil.find_intervals(chase, True)

        chase_intervals = socialutil.merge_intervals(chase_intervals, maximum_gap_frames)
        chase_intervals = [
            (start, stop)
            for start, stop in chase_intervals
            if stop - start >= minimum_duration_frames
        ]

        for chase_interval in chase_intervals:

            # The chase interval is relative to the track_relationship start frame.
            # Let's change it so that that it's absolute frame index
            chase_start, chase_stop = chase_interval
            chase_start += track_relationship['start_frame']
            chase_stop += track_relationship['start_frame']

            # figure out who is chasing and who is being chased
            chaser_track_id = -1
            chasee_track_id = -1

            track1_chaser_proportion, track2_chaser_proportion = socialutil.detect_chase_direction(
                track_relationship, chase_interval, maximum_chase_norm_of_deviation)
            if track1_chaser_proportion >= maximum_chase_agreement:
                chaser_track_id = track_relationship['track1']['track_id']
                chasee_track_id = track_relationship['track2']['track_id']
            elif track2_chaser_proportion >= maximum_chase_agreement:
                chaser_track_id = track_relationship['track2']['track_id']
                chasee_track_id = track_relationship['track1']['track_id']

            if chaser_track_id != -1:
                yield {
                    'chaser_track_id': chaser_track_id,
                    'chasee_track_id': chasee_track_id,
                    'start_frame': chase_start,
                    'stop_frame_exclu': chase_stop,
                }


def detect_point_contact_events(
        track_relationships,
        point_index1, point_index2,
        maximum_distance_px, minimum_duration_frames, maximum_gap_frames):

    for track_relationship in track_relationships:
        proximal_frames = socialutil.detect_point_proximity(
            track_relationship, point_index1, point_index2)

        proximity_intervals = socialutil.find_intervals(proximal_frames, True)

        proximity_intervals = socialutil.merge_intervals(proximity_intervals, maximum_gap_frames)
        proximity_intervals = [
            (start, stop)
            for start, stop in proximity_intervals
            if stop - start >= minimum_duration_frames
        ]

        for proximity_interval in proximity_intervals:

            # The interval is relative to the track_relationship start frame.
            # Let's change it so that that it's absolute frame index
            contact_start, contact_stop = proximity_interval
            contact_start += track_relationship['start_frame']
            contact_stop += track_relationship['start_frame']

            yield {
                'track1_id': track_relationship['track1']['track_id'],
                'track2_id': track_relationship['track2']['track_id'],
                'start_frame': contact_start,
                'stop_frame_exclu': contact_stop,
            }


def detect_oral_oral_contact_events(track_relationships, social_config):

    oral_oral_config = social_config['behavior']['oral_oral_contact']

    pose_config = social_config['pose']
    fps = pose_config['frames_per_sec']
    pixels_per_cm = pose_config['pixels_per_cm']

    maximum_distance_px = oral_oral_config['maximum_distance_cm'] * pixels_per_cm
    minimum_duration_frames = oral_oral_config['minimum_duration_sec'] * fps
    maximum_gap_frames = oral_oral_config['maximum_gap_sec'] * fps

    return detect_point_contact_events(
        track_relationships,
        socialutil.NOSE_INDEX, socialutil.NOSE_INDEX,
        maximum_distance_px, minimum_duration_frames, maximum_gap_frames)


def detect_oral_genital_contact_events(track_relationships, social_config):

    oral_genital_config = social_config['behavior']['oral_genital_contact']

    pose_config = social_config['pose']
    fps = pose_config['frames_per_sec']
    pixels_per_cm = pose_config['pixels_per_cm']

    maximum_distance_px = oral_genital_config['maximum_distance_cm'] * pixels_per_cm
    minimum_duration_frames = oral_genital_config['minimum_duration_sec'] * fps
    maximum_gap_frames = oral_genital_config['maximum_gap_sec'] * fps

    return itertools.chain(

        detect_point_contact_events(
            track_relationships,
            socialutil.NOSE_INDEX, socialutil.BASE_TAIL_INDEX,
            maximum_distance_px, minimum_duration_frames, maximum_gap_frames),

        detect_point_contact_events(
            track_relationships,
            socialutil.BASE_TAIL_INDEX, socialutil.NOSE_INDEX,
            maximum_distance_px, minimum_duration_frames, maximum_gap_frames),

    )


def gen_social_stats(net_file_name, pose_file_name, social_config):
    instance_tracks = gen_instance_tracks(pose_file_name, social_config)
    track_relationships = list(socialutil.calc_track_relationships(
        sorted(instance_tracks.values(), key=lambda track: track['start_frame'])))
    all_chases = list(detect_chase_events(track_relationships, social_config))
    all_oral_oral = list(detect_oral_oral_contact_events(track_relationships, social_config))
    all_oral_genital = list(detect_oral_genital_contact_events(track_relationships, social_config))

    return {
        'network_filename': net_file_name,
        'chases': all_chases,
        'oral_oral_contact': all_oral_oral,
        'oral_genital_contact': all_oral_genital,
    }


def _gen_social_stats(data_file_tuple, social_config):
    net_file_name, pose_file_name = data_file_tuple
    return gen_social_stats(net_file_name, pose_file_name, social_config)


def gen_all_social_stats(data_file_names, social_config, num_procs):

    gen_social_stats_partial = functools.partial(
        _gen_social_stats,
        social_config=social_config,
    )

    if num_procs == 1:
        for data_file_tuple in data_file_names:
            video_stats = gen_social_stats_partial(data_file_tuple)
            if video_stats is not None:
                yield video_stats
    else:
        with mp.Pool(num_procs) as p:
            for video_stats in p.imap_unordered(gen_social_stats_partial, data_file_names):
                if video_stats is not None:
                    yield video_stats


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
    parser.add_argument(
        '--num-procs',
        help='the number of processes to use',
        default=12,
        type=int,
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
                pose_file_name = os.path.join(args.root_dir, data_file_base + '_pose_est_v3.h5')
                data_file_names.append((net_file_name, pose_file_name))

    outdir = os.path.dirname(args.out_file)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    with h5py.File(args.out_file, 'w') as social_h5:
        for video_stats in gen_all_social_stats(data_file_names, social_config, args.num_procs):

            escaped_file_name = urlparse.quote(video_stats['network_filename'], safe='')
            vid_grp = social_h5.create_group(escaped_file_name)

            # save chases to HDF5
            all_chases = video_stats['chases']
            chase_track_ids = np.array(
                [[c['chaser_track_id'], c['chasee_track_id']] for c in all_chases],
                dtype=np.uint32,
            )
            vid_grp['chase_track_ids'] = chase_track_ids

            chase_start_frame = np.array(
                [c['start_frame'] for c in all_chases],
                dtype=np.uint32,
            )
            vid_grp['chase_start_frame'] = chase_start_frame

            chase_frame_count = np.array(
                [c['stop_frame_exclu'] - c['start_frame'] for c in all_chases],
                dtype=np.uint32,
            )
            vid_grp['chase_frame_count'] = chase_frame_count

            # save oral oral contact to HDF5
            all_oral_oral_contact = video_stats['oral_oral_contact']
            oral_oral_track_ids = np.array(
                [[ooc['track1_id'], ooc['track2_id']] for ooc in all_oral_oral_contact],
                dtype=np.uint32,
            )
            vid_grp['oral_oral_contact_track_ids'] = oral_oral_track_ids

            oral_oral_start_frame = np.array(
                [ooc['start_frame'] for ooc in all_oral_oral_contact],
                dtype=np.uint32,
            )
            vid_grp['oral_oral_contact_start_frame'] = oral_oral_start_frame

            oral_oral_frame_count = np.array(
                [ooc['stop_frame_exclu'] - ooc['start_frame'] for ooc in all_oral_oral_contact],
                dtype=np.uint32,
            )
            vid_grp['oral_oral_contact_frame_count'] = oral_oral_frame_count

            # save oral genital contact to HDF5
            all_oral_genital_contact = video_stats['oral_genital_contact']
            oral_genital_track_ids = np.array(
                [[ogc['track1_id'], ogc['track2_id']] for ogc in all_oral_genital_contact],
                dtype=np.uint32,
            )
            vid_grp['oral_genital_contact_track_ids'] = oral_genital_track_ids

            oral_genital_start_frame = np.array(
                [ogc['start_frame'] for ogc in all_oral_genital_contact],
                dtype=np.uint32,
            )
            vid_grp['oral_genital_contact_start_frame'] = oral_genital_start_frame

            oral_genital_frame_count = np.array(
                [ogc['stop_frame_exclu'] - ogc['start_frame'] for ogc in all_oral_genital_contact],
                dtype=np.uint32,
            )
            vid_grp['oral_genital_contact_frame_count'] = oral_genital_frame_count

            print('PROCESSED:', video_stats['network_filename'])


if __name__ == "__main__":
    main()
