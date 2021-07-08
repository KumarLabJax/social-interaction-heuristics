import argparse
import csv
import functools
import h5py
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

        return track_dict, len(all_instance_count)


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
                    'chaser_track_id': int(chaser_track_id),
                    'chasee_track_id': int(chasee_track_id),
                    'start_frame': int(chase_start),
                    'stop_frame_exclu': int(chase_stop),
                }


def detect_point_contact_events(
        track_relationships,
        point_index1, point_index2,
        maximum_distance_px, minimum_duration_frames, maximum_gap_frames,
        approach_dist_px, approach_frames_before_contact):

    for track_relationship in track_relationships:
        proximal_frames = socialutil.detect_point_proximity(
            track_relationship, point_index1, point_index2, maximum_distance_px)

        proximity_intervals = socialutil.find_intervals(proximal_frames, True)

        proximity_intervals = socialutil.merge_intervals(proximity_intervals, maximum_gap_frames)
        proximity_intervals = [
            (start, stop)
            for start, stop in proximity_intervals
            if stop - start >= minimum_duration_frames
        ]

        for proximity_interval in proximity_intervals:

            # filter out any contact events that were not preceeded by an approach
            contact_start, contact_stop = proximity_interval

            approach_start_index = max(0, contact_start - approach_frames_before_contact)
            track_distances = track_relationship['track_distances']
            contact_start_distance = track_distances[contact_start]
            approach_max_distance = track_distances[approach_start_index : contact_start + 1].max()

            if approach_max_distance - contact_start_distance >= approach_dist_px:
                # The interval is relative to the track_relationship start frame.
                # Let's change it so that that it's absolute frame index
                contact_start += track_relationship['start_frame']
                contact_stop += track_relationship['start_frame']

                yield {
                    'track1_id': int(track_relationship['track1']['track_id']),
                    'track2_id': int(track_relationship['track2']['track_id']),
                    'start_frame': int(contact_start),
                    'stop_frame_exclu': int(contact_stop),
                }


def detect_oral_oral_contact_events(track_relationships, social_config):

    oral_oral_config = social_config['behavior']['oral_oral_contact']

    pose_config = social_config['pose']
    fps = pose_config['frames_per_sec']
    pixels_per_cm = pose_config['pixels_per_cm']

    maximum_distance_px = oral_oral_config['maximum_distance_cm'] * pixels_per_cm
    minimum_duration_frames = oral_oral_config['minimum_duration_sec'] * fps
    maximum_gap_frames = oral_oral_config['maximum_gap_sec'] * fps

    approach_dist_px = oral_oral_config['approach_dist_cm'] * pixels_per_cm
    approach_frames_before_contact = oral_oral_config['approach_secs_before_contact'] * fps

    return detect_point_contact_events(
        track_relationships,
        socialutil.NOSE_INDEX, socialutil.NOSE_INDEX,
        maximum_distance_px, minimum_duration_frames, maximum_gap_frames,
        approach_dist_px, approach_frames_before_contact)


def detect_oral_genital_contact_events(track_relationships, social_config):

    oral_genital_config = social_config['behavior']['oral_genital_contact']

    pose_config = social_config['pose']
    fps = pose_config['frames_per_sec']
    pixels_per_cm = pose_config['pixels_per_cm']

    maximum_distance_px = oral_genital_config['maximum_distance_cm'] * pixels_per_cm
    minimum_duration_frames = oral_genital_config['minimum_duration_sec'] * fps
    maximum_gap_frames = oral_genital_config['maximum_gap_sec'] * fps

    approach_dist_px = oral_genital_config['approach_dist_cm'] * pixels_per_cm
    approach_frames_before_contact = oral_genital_config['approach_secs_before_contact'] * fps

    return itertools.chain(

        detect_point_contact_events(
            track_relationships,
            socialutil.NOSE_INDEX, socialutil.BASE_TAIL_INDEX,
            maximum_distance_px, minimum_duration_frames, maximum_gap_frames,
            approach_dist_px, approach_frames_before_contact),

        detect_point_contact_events(
            track_relationships,
            socialutil.BASE_TAIL_INDEX, socialutil.NOSE_INDEX,
            maximum_distance_px, minimum_duration_frames, maximum_gap_frames,
            approach_dist_px, approach_frames_before_contact),

    )


def detect_oral_ear_contact_events(track_relationships, social_config):

    oral_ear_config = social_config['behavior']['oral_ear_contact']

    pose_config = social_config['pose']
    fps = pose_config['frames_per_sec']
    pixels_per_cm = pose_config['pixels_per_cm']

    maximum_distance_px = oral_ear_config['maximum_distance_cm'] * pixels_per_cm
    minimum_duration_frames = oral_ear_config['minimum_duration_sec'] * fps
    maximum_gap_frames = oral_ear_config['maximum_gap_sec'] * fps

    approach_dist_px = oral_ear_config['approach_dist_cm'] * pixels_per_cm
    approach_frames_before_contact = oral_ear_config['approach_secs_before_contact'] * fps

    return itertools.chain(

        detect_point_contact_events(
            track_relationships,
            socialutil.NOSE_INDEX, socialutil.LEFT_EAR_INDEX,
            maximum_distance_px, minimum_duration_frames, maximum_gap_frames,
            approach_dist_px, approach_frames_before_contact),

        detect_point_contact_events(
            track_relationships,
            socialutil.NOSE_INDEX, socialutil.RIGHT_EAR_INDEX,
            maximum_distance_px, minimum_duration_frames, maximum_gap_frames,
            approach_dist_px, approach_frames_before_contact),

        detect_point_contact_events(
            track_relationships,
            socialutil.LEFT_EAR_INDEX, socialutil.NOSE_INDEX,
            maximum_distance_px, minimum_duration_frames, maximum_gap_frames,
            approach_dist_px, approach_frames_before_contact),

        detect_point_contact_events(
            track_relationships,
            socialutil.RIGHT_EAR_INDEX, socialutil.NOSE_INDEX,
            maximum_distance_px, minimum_duration_frames, maximum_gap_frames,
            approach_dist_px, approach_frames_before_contact),

    )


def detect_approach_events(track_relationships, social_config):
    approach_config = social_config['behavior']['approach']

    pose_config = social_config['pose']
    fps = pose_config['frames_per_sec']
    pixels_per_cm = pose_config['pixels_per_cm']

    minimum_pre_approach_distance_px = approach_config['minimum_pre_approach_distance_cm'] * pixels_per_cm
    maximum_arrival_distance_px = approach_config['maximum_arrival_distance_cm'] * pixels_per_cm
    maximum_approach_duration_frames = approach_config['maximum_approach_duration_sec'] * fps
    maximum_still_speed_px_frame = approach_config['maximum_still_speed_cm_sec'] * pixels_per_cm / fps

    det_approach = functools.partial(
            socialutil.detect_approach_intervals,
            minimum_pre_approach_distance_px=minimum_pre_approach_distance_px,
            maximum_arrival_distance_px=maximum_arrival_distance_px,
            maximum_approach_duration_frames=maximum_approach_duration_frames,
            maximum_still_speed_px_frame=maximum_still_speed_px_frame)

    return itertools.chain.from_iterable(det_approach(t_rel) for t_rel in track_relationships)


def detect_huddles(track_relationships, social_config):

    huddle_config = social_config['behavior']['huddle']

    pose_config = social_config['pose']
    fps = pose_config['frames_per_sec']
    pixels_per_cm = pose_config['pixels_per_cm']

    maximum_distance_px = huddle_config['maximum_distance_cm'] * pixels_per_cm
    minimum_duration_frames = huddle_config['minimum_duration_sec'] * fps
    maximum_displacement_px = huddle_config['maximum_displacement_cm'] * pixels_per_cm
    maximum_gap_merge_frames = huddle_config['maximum_gap_merge_sec'] * fps

    for track_relationship in track_relationships:

        curr_huddles = socialutil.detect_pairwise_proximity_intervals(
                track_relationship,
                maximum_distance_px,
                minimum_duration_frames,
                maximum_displacement_px,
                maximum_gap_merge_frames)

        for huddle_start, huddle_stop in curr_huddles:
            yield {
                'track1_id': int(track_relationship['track1']['track_id']),
                'track2_id': int(track_relationship['track2']['track_id']),
                'start_frame': int(huddle_start),
                'stop_frame_exclu': int(huddle_stop),
            }


def detect_watching(track_relationships, social_config):

    watching_config = social_config['behavior']['watching']

    pose_config = social_config['pose']
    fps = pose_config['frames_per_sec']
    pixels_per_cm = pose_config['pixels_per_cm']

    maximum_gaze_offset_deg = watching_config['maximum_gaze_offset_deg']
    minimum_distance_px = watching_config['minimum_distance_cm'] * pixels_per_cm
    maximum_distance_px = watching_config['maximum_distance_cm'] * pixels_per_cm
    minimum_duration_frames = watching_config['minimum_duration_sec'] * fps
    maximum_gap_merge_frames = watching_config['maximum_gap_merge_sec'] * fps

    return itertools.chain.from_iterable(
        socialutil.detect_watch_intervals(
            tr,
            maximum_gaze_offset_deg,
            minimum_distance_px,
            maximum_distance_px,
            minimum_duration_frames,
            maximum_gap_merge_frames)
        for tr in track_relationships
    )


def detect_proximity(track_relationships, social_config, section_name):
    proximity_config = social_config['behavior'][section_name]

    pose_config = social_config['pose']
    fps = pose_config['frames_per_sec']
    pixels_per_cm = pose_config['pixels_per_cm']

    maximum_distance_px = proximity_config['maximum_distance_cm'] * pixels_per_cm
    minimum_duration_frames = 0
    maximum_displacement_px = None
    maximum_gap_merge_frames = proximity_config['maximum_gap_merge_sec'] * fps

    for track_relationship in track_relationships:

        curr_proximity_intervals = socialutil.detect_pairwise_proximity_intervals(
                track_relationship,
                maximum_distance_px,
                minimum_duration_frames,
                maximum_displacement_px,
                maximum_gap_merge_frames)

        for inter_start, inter_stop in curr_proximity_intervals:
            yield {
                'track1_id': int(track_relationship['track1']['track_id']),
                'track2_id': int(track_relationship['track2']['track_id']),
                'start_frame': int(inter_start),
                'stop_frame_exclu': int(inter_stop),
            }


def distance_traveled_per_track(tracks, social_config):
    # extract some config stuff
    pose_config = social_config['pose']

    fps = pose_config['frames_per_sec']
    pixels_per_cm = pose_config['pixels_per_cm']

    dist_traveled_conf = social_config['behavior']['distance_traveled']
    still_displacement_threshold_px = dist_traveled_conf['still_displacement_threshold_cm'] * pixels_per_cm
    still_time_threshold_frames = dist_traveled_conf['still_time_threshold_sec'] * fps

    for track in tracks:
        dist_traveled_px = socialutil.calc_track_distance_traveled(
            track,
            still_displacement_threshold_px,
            still_time_threshold_frames,
        )

        yield {
            'track_id': int(track['track_id']),
            'distance_traveled_cm': float(dist_traveled_px / pixels_per_cm),
        }


def gen_social_stats(net_file_name, pose_file_name, social_config):
    print('working on:', net_file_name)
    instance_tracks, frame_count = gen_instance_tracks(pose_file_name, social_config)
    all_distance_traveled = list(distance_traveled_per_track(instance_tracks.values(), social_config))
    track_relationships = list(socialutil.calc_track_relationships(
        sorted(instance_tracks.values(), key=lambda track: track['start_frame'])))
    all_contact = list(detect_proximity(track_relationships, social_config, 'contact'))
    all_close = list(detect_proximity(track_relationships, social_config, 'close'))
    all_watching = list(detect_watching(track_relationships, social_config))
    all_huddles = list(detect_huddles(track_relationships, social_config))
    all_approaches = list(detect_approach_events(track_relationships, social_config))
    all_chases = list(detect_chase_events(track_relationships, social_config))
    all_oral_oral = list(detect_oral_oral_contact_events(track_relationships, social_config))
    all_oral_genital = list(detect_oral_genital_contact_events(track_relationships, social_config))
    all_oral_ear = list(detect_oral_ear_contact_events(track_relationships, social_config))

    return {
        'network_filename': net_file_name,
        'distance_traveled': all_distance_traveled,
        'chases': all_chases,
        'oral_oral_contact': all_oral_oral,
        'oral_genital_contact': all_oral_genital,
        'oral_ear_contact': all_oral_ear,
        'approaches': all_approaches,
        'huddles': all_huddles,
        'watching': all_watching,
        'contact': all_contact,
        'close': all_close,
        'frame_count': frame_count,
    }


def _gen_social_stats(data_file_tuple, social_config):

    net_file_name, pose_file_name = data_file_tuple
    try:
        return gen_social_stats(net_file_name, pose_file_name, social_config)
    except Exception:
        print('ERROR: FAILED TO PROCESS', net_file_name, 'and', pose_file_name)
        return None


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

# share_root=/home/sheppk/smb/labshare/VideoData/MDS_Tests/BTBR_3M_stranger_4day
# python -u gensocialstats.py \
#   --social-config social-config.yaml \
#   --batch-file btbr-vids.txt \
#   --root-dir "${share_root}" \
#   --out-file BTBR_3M_stranger_4day-out-2020-04-22.yaml

# share_root=/home/sheppk/smb/labshare/VideoData/MDS_Tests/B6J_3M_stranger_4day
# python -u gensocialstats.py \
#   --social-config social-config.yaml \
#   --batch-file b6-vids.txt \
#   --root-dir "${share_root}" \
#   --out-file B6_3M_stranger_4day-out-2020-04-22.yaml

# python -u gensocialstats.py \
#   --social-config social-config.yaml \
#   --batch-file btbr-vids.txt \
#   --root-dir ~/smb/labshare/VideoData/MDS_Tests/BTBR_3M_stranger_4day \
#   --out-file BTBR_3M_stranger_4day-out-2020-04-22.yaml
# python -u gensocialstats.py \
#   --social-config social-config.yaml \
#   --batch-file b6-vids.txt \
#   --root-dir ~/smb/labshare/VideoData/MDS_Tests/B6J_3M_stranger_4day \
#   --out-file B6_3M_stranger_4day-out-2020-04-22.yaml

# python -u gensocialstats.py \
#   --social-config social-config.yaml \
#   --batch-file ucsd-vids.txt \
#   --root-dir ~/smb/labshare \
#   --out-file UCSD-out-2020-06-15.yaml

# python -u gensocialstats.py \
#   --social-config social-config.yaml \
#   --batch-file ucsd-vids-2020-08-04.txt \
#   --root-dir ~/smb/labshare \
#   --out-file UCSD-out-2020-08-04.yaml

# python -u gensocialstats.py \
#   --social-config social-config-2020-08-27.yaml \
#   --batch-file ucsd-vids-2020-08-04.txt \
#   --root-dir ~/smb/labshare \
#   --out-file UCSD-out-2020-08-28.yaml

# BXD:
#   python -u gensocialstats.py \
#       --social-config social-config-2020-08-27.yaml \
#       --batch-file bxd-batch-2020-08-28.txt\
#       --root-dir ~/smb/labshare \
#       --out-file bxd-out-2020-08-28.yaml


# B2B vs CBAX2B - using B6J vs BTBR Three Male Stranger, Four Day Social Interaction
#   ./scripts/find-b2b-b6-vids.sh > ./data/B2B_B6_3M_stranger_4day-out-2020-11-12-batch.txt
#   share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar'
#   python -u gensocialstats.py \
#       --num-procs 3 \
#       --social-config social-config-2020-08-27.yaml \
#       --batch-file ./data/B2B_B6_3M_stranger_4day-out-2020-11-12-batch.txt \
#       --root-dir "${share_root}" \
#       --out-file ./data/B2B_B6_3M_stranger_4day-out-2020-11-12-social.yaml
#   ./scripts/find-cbax2-b6-vids.sh > ./data/CBAX2_B6_3M_stranger_4day-out-2020-11-12-batch.txt
#   share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar/VideoData/MDS_Tests/B6J_3M_stranger_4day'
#   python -u gensocialstats.py \
#       --num-procs 3 \
#       --social-config social-config-2020-08-27.yaml \
#       --batch-file ./data/CBAX2_B6_3M_stranger_4day-out-2020-11-12-batch.txt \
#       --root-dir "${share_root}" \
#       --out-file ./data/CBAX2_B6_3M_stranger_4day-out-2020-11-12-social.yaml
#   ./scripts/find-b2b-btbr-vids.sh > ./data/B2B_BTBR_3M_stranger_4day-out-2020-11-12-batch.txt
#   ./scripts/find-cbax2-btbr-vids.sh > ./data/CBAX2_BTBR_3M_stranger_4day-out-2020-11-12-batch.txt

# WORKING TO RECOVER MISSING POSES
#   source ~/venvs/ofp/bin/activate
#   share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar'
#   python -u ~/projects/open-field-pipeline/local/postprocverify.py --batch-file data/B2B_ALL_3M_stranger_4day-out-2020-11-12-batch.txt --root "${share_root}" --suffix '_pose_est_v3.h5' > data/B2B_MISSING_3M_stranger_4day-out-2020-11-12-batch.txt
#   python ~/projects/open-field-pipeline/local/verifybatch.py --batch-file data/B2B_MISSING_3M_stranger_4day-out-2020-11-12-batch.txt --src-root "${share_root}" --dest-root ~/sshfs/winterfastscratch/B2B_MISSING_3M_stranger_4day-2020-11-12

# Reboot: B2B vs CBAX2B vs Building B6 - using B6J vs BTBR Three Male Stranger, Four Day Social Interaction
#
#   ./scripts/find-b2b-b6-poses.sh > ./data/B2B_B6_3M_stranger_4day-out-2020-12-09-batch.txt
#   share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar'
#   python -u gensocialstats.py \
#       --num-procs 3 \
#       --social-config social-config-2020-08-27.yaml \
#       --batch-file ./data/B2B_B6_3M_stranger_4day-out-2020-12-09-batch.txt \
#       --root-dir "${share_root}" \
#       --out-file ./data/B2B_B6_3M_stranger_4day-out-2020-12-09-social.yaml
#
#   ./scripts/find-cbax2-b6-poses.sh > ./data/CBAX2_B6_3M_stranger_4day-out-2020-12-09-batch.txt
#   share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar/VideoData/MDS_Tests/B6J_3M_stranger_4day'
#   python -u gensocialstats.py \
#       --num-procs 3 \
#       --social-config social-config-2020-08-27.yaml \
#       --batch-file ./data/CBAX2_B6_3M_stranger_4day-out-2020-12-09-batch.txt \
#       --root-dir "${share_root}" \
#       --out-file ./data/CBAX2_B6_3M_stranger_4day-out-2020-12-09-social.yaml
#
#   ./scripts/find-b6-b6-poses.sh > ./data/B6_B6_3M_stranger_4day-out-2021-05-13-batch.txt
#   share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar/VideoData/MDS_Tests/B6J_3M_stranger_4day'
#   python -u gensocialstats.py \
#       --num-procs 6 \
#       --social-config social-config-2020-08-27.yaml \
#       --batch-file ./data/B6_B6_3M_stranger_4day-out-2021-05-13-batch.txt \
#       --root-dir "${share_root}" \
#       --out-file ./data/B6_B6_3M_stranger_4day-out-2021-05-13-social.yaml
#
#   ./scripts/find-b2b-btbr-poses.sh > ./data/B2B_BTBR_3M_stranger_4day-out-2020-12-09-batch.txt
#   share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar'
#   python -u gensocialstats.py \
#       --num-procs 3 \
#       --social-config social-config-2020-08-27.yaml \
#       --batch-file ./data/B2B_BTBR_3M_stranger_4day-out-2020-12-09-batch.txt \
#       --root-dir "${share_root}" \
#       --out-file ./data/B2B_BTBR_3M_stranger_4day-out-2020-12-09-social.yaml
#
#   ./scripts/find-cbax2-btbr-poses.sh > ./data/CBAX2_BTBR_3M_stranger_4day-out-2020-12-09-batch.txt
#   share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar/VideoData/MDS_Tests/BTBR_3M_stranger_4day'
#   python -u gensocialstats.py \
#       --num-procs 3 \
#       --social-config social-config-2020-08-27.yaml \
#       --batch-file ./data/CBAX2_BTBR_3M_stranger_4day-out-2020-12-09-batch.txt \
#       --root-dir "${share_root}" \
#       --out-file ./data/CBAX2_BTBR_3M_stranger_4day-out-2020-12-09-social.yaml
#
#   ./scripts/find-b6-btbr-poses.sh > ./data/B6_BTBR_3M_stranger_4day-out-2021-05-13-batch.txt
#   share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar/VideoData/MDS_Tests/BTBR_3M_stranger_4day'
#   python -u gensocialstats.py \
#       --num-procs 6 \
#       --social-config social-config-2020-08-27.yaml \
#       --batch-file ./data/B6_BTBR_3M_stranger_4day-out-2021-05-13-batch.txt \
#       --root-dir "${share_root}" \
#       --out-file ./data/B6_BTBR_3M_stranger_4day-out-2021-05-13-social.yaml

# AD models - PS19
#   root_dir='/media/sheppk/TOSHIBA EXT/AD-models-PS19-poses'
#   python -u gensocialstats.py \
#       --num-procs 12 \
#       --social-config social-config-2020-08-27.yaml \
#       --batch-file "${root_dir}/batch.txt" \
#       --root-dir "${root_dir}" \
#       --out-file ./data/AD_models-PS19-out-2021-03-04-social.yaml

#   share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar/VideoData/MDS_Tests/BTBR_3M_stranger_4day'
#   python -u gensocialstats.py \
#       --num-procs 6 \
#       --social-config social-config.yaml \
#       --batch-file ./data/B6_BTBR_3M_stranger_4day-out-2021-05-13-batch.txt \
#       --root-dir "${share_root}" \
#       --out-file ./data/B6_BTBR_3M_stranger_4day-out-2021-06-10-social.yaml

# 2nd Reboot: B2B vs CBAX2B vs Building B6 - using B6J vs BTBR Three Male Stranger, Four Day Social Interaction
#
#   share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar/VideoData/MDS_Tests/B6J_3M_stranger_4day'
#   python -u gensocialstats.py \
#       --num-procs 3 \
#       --social-config social-config.yaml \
#       --batch-file ./data/B2B_B6_3M_stranger_4day-out-2020-12-09-batch.txt \
#       --root-dir "${share_root}" \
#       --out-file ./data/B2B_B6_3M_stranger_4day-out-2021-06-11-social.yaml
#
#   share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar/VideoData/MDS_Tests/B6J_3M_stranger_4day'
#   python -u gensocialstats.py \
#       --num-procs 3 \
#       --social-config social-config.yaml \
#       --batch-file ./data/CBAX2_B6_3M_stranger_4day-out-2020-12-09-batch.txt \
#       --root-dir "${share_root}" \
#       --out-file ./data/CBAX2_B6_3M_stranger_4day-out-2021-06-11-social.yaml
#
#   share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar/VideoData/MDS_Tests/B6J_3M_stranger_4day'
#   python -u gensocialstats.py \
#       --num-procs 3 \
#       --social-config social-config.yaml \
#       --batch-file ./data/B6_B6_3M_stranger_4day-out-2021-05-13-batch.txt \
#       --root-dir "${share_root}" \
#       --out-file ./data/B6_B6_3M_stranger_4day-out-2021-06-11-social.yaml
#
#   share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar/VideoData/MDS_Tests/BTBR_3M_stranger_4day'
#   python -u gensocialstats.py \
#       --num-procs 3 \
#       --social-config social-config.yaml \
#       --batch-file ./data/B2B_BTBR_3M_stranger_4day-out-2020-12-09-batch.txt \
#       --root-dir "${share_root}" \
#       --out-file ./data/B2B_BTBR_3M_stranger_4day-out-2021-06-11-social.yaml
#
#   share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar/VideoData/MDS_Tests/BTBR_3M_stranger_4day'
#   python -u gensocialstats.py \
#       --num-procs 3 \
#       --social-config social-config.yaml \
#       --batch-file ./data/CBAX2_BTBR_3M_stranger_4day-out-2020-12-09-batch.txt \
#       --root-dir "${share_root}" \
#       --out-file ./data/CBAX2_BTBR_3M_stranger_4day-out-2021-06-11-social.yaml
#
#   share_root='/run/user/1000/gvfs/smb-share:server=bht2stor.jax.org,share=vkumar/VideoData/MDS_Tests/BTBR_3M_stranger_4day'
#   python -u gensocialstats.py \
#       --num-procs 3 \
#       --social-config social-config.yaml \
#       --batch-file ./data/B6_BTBR_3M_stranger_4day-out-2021-05-13-batch.txt \
#       --root-dir "${share_root}" \
#       --out-file ./data/B6_BTBR_3M_stranger_4day-out-2021-06-11-social.yaml

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
                pose_file_name = os.path.join(args.root_dir, data_file_base + '_pose_est_v4.h5')
                if not os.path.exists(pose_file_name):
                    pose_file_name = os.path.join(args.root_dir, data_file_base + '_pose_est_v3.h5')
                data_file_names.append((net_file_name, pose_file_name))

    outdir = os.path.dirname(args.out_file)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    with open(args.out_file, 'w') as social_yaml:
        yaml.safe_dump_all(
            gen_all_social_stats(data_file_names, social_config, args.num_procs),
            social_yaml)


if __name__ == "__main__":
    main()
