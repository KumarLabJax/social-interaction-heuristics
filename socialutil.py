import itertools
import math
import numpy as np
from shapely.geometry import MultiPoint

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


def xy_dist(pt1, pt2):
    x_diff = pt2['x_pos'] - pt1['x_pos']
    y_diff = pt2['y_pos'] - pt1['y_pos']

    return math.sqrt(x_diff ** 2 + y_diff ** 2)


def mean_point_distance_travelled(track, start_index, stop_index_exclu, point_indexes=None):
    prev_point_info = dict()
    point_distance_sum = 0
    point_distance_denom = 0

    track_points = track['points']
    track_point_masks = track['point_masks']

    for pose_index in range(start_index, stop_index_exclu):
        curr_points = track_points[pose_index]
        curr_mask = track_point_masks[pose_index]

        if point_indexes is None:
            point_indexes = list(range(len(curr_mask)))

        for point_index in point_indexes:
            if curr_mask[point_index]:
                curr_point_info = {
                    'y_pos': float(curr_points[point_index, 0]),
                    'x_pos': float(curr_points[point_index, 1]),
                    'pose_index': pose_index,
                }

                if point_index in prev_point_info:
                    point_distance_sum += xy_dist(curr_point_info, prev_point_info[point_index])
                    point_distance_denom += pose_index - prev_point_info[point_index]['pose_index']

                prev_point_info[point_index] = curr_point_info

    if point_distance_denom == 0:
        return None
    else:
        point_distance_denom /= stop_index_exclu - start_index
        return point_distance_sum / point_distance_denom


def interpolate_interval(track, point_index, pose_start_index, pose_stop_index_exclu):
    """
    Performs a simple linear interpolation for the indicated point from start to stop
    and updates the mask along this interval setting it to True.
    """

    pose_stop_index = pose_stop_index_exclu - 1
    start_yx = track['points'][pose_start_index, point_index, :].astype(np.float32)
    stop_yx = track['points'][pose_stop_index, point_index, :].astype(np.float32)

    diff_yx = stop_yx - start_yx
    interp_denom = pose_stop_index - pose_start_index
    for curr_pose_index in range(pose_start_index + 1, pose_stop_index):
        interp_step = curr_pose_index - pose_start_index
        interp_yx = start_yx + (diff_yx * interp_step / interp_denom)
        track['points'][curr_pose_index, point_index, :] = interp_yx.round()
        track['point_masks'][curr_pose_index, point_index] = True


def interpolate_missing_points(track, maximum_point_distance_px, maximum_mean_point_distance_px):

    """
    Interpolate (y, x) position across all missing point locations that meet
    the given threshold conditions.

    :param track: the track that we're interpolating over
    :param maximum_point_distance_px: if points move farther than this pixel distance we don't
        perform the interpolation
    :param maximum_mean_point_distance_px: if the mean of *all* point distances over this interval
        exceeds this value we don't perform interpolation
    """

    track_point_masks = track['point_masks']

    # first we determine which missing point intervals meet the
    # threshold for interpolation. If they pass the threshold we
    # put them in the interp_intervals list and take care of
    # computing the interpolation later
    interp_intervals = []
    pose_count = len(track_point_masks)
    for pose_index in range(pose_count - 1):
        curr_mask = track_point_masks[pose_index]
        next_mask = track_point_masks[pose_index + 1]
        for point_index in range(len(curr_mask)):
            # if current point exists and next is missing we should
            # attempt an interpolation
            if curr_mask[point_index] and not next_mask[point_index]:

                # try to find the next occurance of this point index
                next_valid_pose_index = None
                for maybe_valid_pose_index in range(pose_index + 2, pose_count):
                    if track_point_masks[maybe_valid_pose_index][point_index]:
                        next_valid_pose_index = maybe_valid_pose_index
                        break

                # we need to do several checks here to make sure that the
                # interval meets our need

                # skip interpolation if we couldn't find a next valid pose
                if next_valid_pose_index is None:
                    continue

                # check the point distance threshold
                if maximum_point_distance_px is not None:
                    point_dist = mean_point_distance_travelled(
                        track,
                        pose_index,
                        next_valid_pose_index + 1,
                        [point_index])
                    assert point_dist is not None

                    if point_dist > maximum_point_distance_px:
                        continue

                # check the mean point distance threshold
                if maximum_mean_point_distance_px is not None:
                    mean_points_dist = mean_point_distance_travelled(
                        track,
                        pose_index,
                        next_valid_pose_index + 1)
                    assert mean_points_dist is not None

                    if mean_points_dist > maximum_mean_point_distance_px:
                        continue

                # we passed all of the checks so we can interpolate this interval
                interp_intervals.append({
                    'point_index': point_index,
                    'pose_start_index': pose_index,
                    'pose_stop_index_exclu': next_valid_pose_index + 1,
                })

    # we now have the list of interpolations that we need to perform, so we
    # just iterate through them and do a seperate interpolation for each interval
    for interp_interval in interp_intervals:
        interpolate_interval(track, **interp_interval)


def build_tracks(all_points, all_confidence, all_instance_count, all_track_id):
    all_points_mask = all_confidence > 0

    track_dict = dict()
    frame_count = len(all_instance_count)
    for frame_index in range(frame_count):
        curr_instance_count = all_instance_count[frame_index]
        curr_track_ids = all_track_id[frame_index, :curr_instance_count]
        for i, curr_track_id in enumerate(curr_track_ids):
            curr_track_points = all_points[frame_index, i, ...]
            curr_track_points_mask = all_points_mask[frame_index, i, :]
            if curr_track_id in track_dict:
                track_dict[curr_track_id]['points'].append(curr_track_points)
                track_dict[curr_track_id]['point_masks'].append(curr_track_points_mask)
            else:
                track_dict[curr_track_id] = {
                    'track_id': curr_track_id,
                    'start_frame': frame_index,
                    'points': [curr_track_points],
                    'point_masks': [curr_track_points_mask],
                }

    for track in track_dict.values():
        track['points'] = np.stack(track['points'])
        track['point_masks'] = np.stack(track['point_masks'])
        track_length = len(track['points'])
        track['length'] = track_length
        track['stop_frame_exclu'] = track_length + track['start_frame']

    return track_dict


def update_track_tensors(track_dict, all_points, all_confidence, all_instance_count, all_track_id):

    """
    takes point and point mask values from the tracks in track_dict and writes them to the
    relevant tensor at the correct frame index. This is useful for instance if we want to write
    positions back to the tensors after an interpolation has been performed for missing points
    """

    for track_id, track in track_dict.items():
        for pose_index in range(len(track['points'])):
            frame_index = pose_index + track['start_frame']
            curr_instance_count = all_instance_count[frame_index]
            curr_track_ids = all_track_id[frame_index, :curr_instance_count]
            curr_track_id_index = curr_track_ids.tolist().index(track_id)

            pose_points = track['points'][pose_index]
            pose_points_mask = track['point_masks'][pose_index]

            all_points[frame_index, curr_track_id_index, ...] = pose_points
            all_confidence[frame_index, curr_track_id_index, :] = pose_points_mask


def find_intervals(vals, search_val):
    cursor = 0
    for key, group in itertools.groupby(vals):
        group_len = sum(1 for _ in group)
        if key == search_val:
            # yield the interval as a tuple (inclusive start and exclusive stop index)
            yield cursor, cursor + group_len

        cursor += group_len


def merge_intervals(intervals, max_gap):
    prev_start = None
    prev_stop = None
    for start, stop in intervals:
        if prev_start is None:
            prev_start = start
            prev_stop = stop
        else:
            gap = start - prev_stop
            if gap <= max_gap:
                prev_stop = stop
            else:
                yield prev_start, prev_stop
                prev_start = start
                prev_stop = stop

    if prev_start is not None:
        yield prev_start, prev_stop


def calculate_point_velocities(track):
    """
    calculates velocities (and speeds) for all of the pose points along the track. These
    get added to the track as point_velocities and point_speeds
    """

    track_points = track['points']
    track_point_masks = track['point_masks']
    track_point_velocities = np.zeros(track_points.shape)

    # print(track_points.shape)

    # we need to perform the calculation within valid intervals or the
    # resulting velocities will be incorrect for frames where a point
    # switches from valid to masked out (or vice versa). Otherwise, we
    # could do this using a single call to np.gradient(...)
    num_poses, num_points = track_point_masks.shape
    # print('track_point_masks.shape:', track_point_masks.shape)
    for point_index in range(num_points):
        for interval_start, interval_stop in find_intervals(track_point_masks[:, point_index], True):
            interval_points = track_points[interval_start:interval_stop, point_index, :]
            # print(interval_points.shape)
            if len(interval_points) > 1:
                interval_velocities = np.gradient(interval_points, axis=-2)
                track_point_velocities[interval_start:interval_stop, point_index, :] = interval_velocities

    track['point_velocities'] = track_point_velocities
    track['point_speeds'] = np.linalg.norm(track_point_velocities, axis=-1)

    mean_point_velocities = []
    # print("MEAN POINT VEL:")
    for pose_index in range(num_poses):
        curr_mask = track_point_masks[pose_index, :-2]
        curr_velocities = track_point_velocities[pose_index, :-2]
        mean_velocity = curr_velocities[curr_mask].mean(axis=-2)

        # print(mean_velocity)
        mean_point_velocities.append(mean_velocity)

    mean_point_velocities = np.stack(mean_point_velocities)
    mean_point_speeds = np.linalg.norm(mean_point_velocities, axis=-1)

    track['mean_point_velocities'] = mean_point_velocities
    track['mean_point_speeds'] = mean_point_speeds

    # print('mean_point_velocities:', mean_point_velocities)


def calculate_convex_hulls(track):
    track_points = track['points']
    track_point_masks = track['point_masks']
    track['convex_hulls'] = list(gen_convex_hulls(track_points, track_point_masks))
    track['centroids'] = np.stack([np.array(ch.centroid) for ch in track['convex_hulls']])


def gen_convex_hulls(points, point_masks):
    for pose_index in range(len(points)):
        curr_points = points[pose_index, :-2, :]
        curr_mask = point_masks[pose_index, :-2]
        curr_shape = MultiPoint(curr_points[curr_mask, :]).convex_hull

        yield curr_shape


def calc_centroid_offset_unit_vectors(centroids1, centroids2):

    centroid_offsets = centroids2 - centroids1
    centroid_offset_norms = np.linalg.norm(centroid_offsets, axis=-1)
    centroid_offset_unit_vectors = centroid_offsets / centroid_offset_norms[..., np.newaxis]

    return centroid_offset_unit_vectors


def calc_track_relationships(tracks):
    num_tracks = len(tracks)
    for i in range(num_tracks):
        for j in range(i + 1, num_tracks):
            relationship = calc_track_relationship(tracks[i], tracks[j])
            if relationship is None:
                break
            else:
                yield relationship


def calc_track_relationship(track1, track2):
    """
    calculate some metrics for the overlapping frames of the two given tracks. If
    there is no overlap None is returned instead
    """

    overlap_start_frame = max(track1['start_frame'], track2['start_frame'])
    overlap_stop_frame_exclu = min(track1['stop_frame_exclu'], track2['stop_frame_exclu'])
    overlap_length = overlap_stop_frame_exclu - overlap_start_frame

    if overlap_stop_frame_exclu <= overlap_start_frame:
        return None

    else:
        # the the points from track1 and track2 but only for frames
        # where they overlap and ignore midtail & tiptail points
        pose_start1 = overlap_start_frame - track1['start_frame']
        pose_stop1 = pose_start1 + overlap_length
        overlap_hulls1 = track1['convex_hulls'][pose_start1:pose_stop1]
        overlap_centroids1 = track1['centroids'][pose_start1:pose_stop1, :]

        pose_start2 = overlap_start_frame - track2['start_frame']
        pose_stop2 = pose_start2 + overlap_length
        overlap_hulls2 = track2['convex_hulls'][pose_start2:pose_stop2]
        overlap_centroids2 = track2['centroids'][pose_start2:pose_stop2, :]

        track_distances = np.array([
            shape1.distance(shape2) for shape1, shape2
            in zip(overlap_hulls1, overlap_hulls2)
        ])

        track_offset_unit_vectors = calc_centroid_offset_unit_vectors(overlap_centroids1, overlap_centroids2)

        track_relationship = {
            'track1': track1,
            'track1_start_pose': pose_start1,
            'track1_stop_pose_exclu': pose_stop1,
            # 'track1_mean_speeds': track1['mean_point_speeds'][pose_start1:pose_stop1],
            # 'track1_mean_velocities': track1['mean_point_velocities'][pose_start1:pose_stop1, :],

            'track2': track2,
            'track2_start_pose': pose_start2,
            'track2_stop_pose_exclu': pose_stop2,
            # 'track2_mean_speeds': track2['mean_point_speeds'][pose_start2:pose_stop2],
            # 'track2_mean_velocities': track2['mean_point_velocities'][pose_start2:pose_stop2, :],

            'start_frame': overlap_start_frame,
            'stop_frame_exclu': overlap_stop_frame_exclu,

            'track_distances': track_distances,
            'track_offset_unit_vectors': track_offset_unit_vectors,
        }

        return track_relationship


def detect_chase(track_relationship, maximum_distance_px, minimum_speed_px_frame):

    track1 = track_relationship['track1']
    track1_start = track_relationship['track1_start_pose']
    track1_stop = track_relationship['track1_stop_pose_exclu']
    track1_speed = track1['mean_point_speeds'][track1_start:track1_stop]
    track1_fast_enough = track1_speed >= minimum_speed_px_frame

    track2 = track_relationship['track2']
    track2_start = track_relationship['track2_start_pose']
    track2_stop = track_relationship['track2_stop_pose_exclu']
    track2_speed = track2['mean_point_speeds'][track2_start:track2_stop]
    track2_fast_enough = track2_speed >= minimum_speed_px_frame

    close_enough = track_relationship['track_distances'] <= maximum_distance_px

    all_enough = np.stack([track1_fast_enough, track2_fast_enough, close_enough]).all(axis=0)

    return all_enough


def detect_point_proximity(
        track_relationship,
        point_index1, point_index2,
        maximum_distance_px):

    track1 = track_relationship['track1']
    track1_start = track_relationship['track1_start_pose']
    track1_stop = track_relationship['track1_stop_pose_exclu']
    track1_points = track1['points'][track1_start:track1_stop, point_index1, :]
    masks1 = track1['point_masks'][track1_start:track1_stop, point_index1]

    track2 = track_relationship['track2']
    track2_start = track_relationship['track2_start_pose']
    track2_stop = track_relationship['track2_stop_pose_exclu']
    track2_points = track2['points'][track2_start:track2_stop, point_index2, :]
    masks2 = track2['point_masks'][track2_start:track2_stop, point_index2]

    point_offsets = track2_points - track1_points
    point_offset_dists = np.linalg.norm(point_offsets, axis=-1)
    points_close_enough = point_offset_dists <= maximum_distance_px

    points_valid_and_close = np.stack([masks1, masks2, points_close_enough]).all(axis=0)

    return points_valid_and_close


def detect_chase_direction(track_relationship, chase_interval, chase_max_norm_of_deviation):
    """
    For each frame in the chase_interval this function determines chase
    direction for the given track_relationship. This is returned as a
    boolean array where True indicates that track1 is the chaser and
    False indicates that track2 is the chaser
    """

    chase_start, chase_stop = chase_interval
    chase_len = chase_stop - chase_start

    # for track 1 get the movement direction vectors (scaled to norm of 1)
    track1_chase_start_pose = track_relationship['track1_start_pose'] + chase_start
    track1_chase_stop_pose = track1_chase_start_pose + chase_len
    track1 = track_relationship['track1']
    track1_velocity = track1['mean_point_velocities'][track1_chase_start_pose:track1_chase_stop_pose, :]
    track1_speed = track1['mean_point_speeds'][track1_chase_start_pose:track1_chase_stop_pose]
    track1_speed[track1_speed == 0] = 1     # prevent div by zero
    track1_movement_direction = track1_velocity / track1_speed[..., np.newaxis]

    # for track 2 get the movement direction vectors (scaled to norm of 1)
    track2_chase_start_pose = track_relationship['track2_start_pose'] + chase_start
    track2_chase_stop_pose = track2_chase_start_pose + chase_len
    track2 = track_relationship['track2']
    track2_velocity = track2['mean_point_velocities'][track2_chase_start_pose:track2_chase_stop_pose, :]
    track2_speed = track2['mean_point_speeds'][track2_chase_start_pose:track2_chase_stop_pose]
    track2_speed[track2_speed == 0] = 1     # prevent div by zero
    track2_movement_direction = track2_velocity / track2_speed[..., np.newaxis]

    offset_unit_vectors = track_relationship['track_offset_unit_vectors'][chase_start:chase_stop]

    # after summing the offset vector and movement vector comparing the size of
    # the norm should give us the chase direction
    track1_chase_norms = np.linalg.norm(track1_movement_direction + offset_unit_vectors, axis=-1)
    track1_chaser_proportion = np.mean(track1_chase_norms >= chase_max_norm_of_deviation)

    track2_chase_norms = np.linalg.norm(track2_movement_direction - offset_unit_vectors, axis=-1)
    track2_chaser_proportion = np.mean(track2_chase_norms >= chase_max_norm_of_deviation)

    return track1_chaser_proportion, track2_chaser_proportion


def norm_of_deviation(theta_deg):
    """
    This function considers two unit vectors. The angle between these
    vectors is the given theta. We calculate and return the norm of the sum
    of these two vectors. This norm is useful for thresholding our
    chase behavior because the norm will increase with a smaller
    absolute value of theta and decrease with a larger absolute value.
    """
    theta_rad = math.radians(theta_deg)
    vec = np.array([1.0 + math.cos(theta_rad), math.sin(theta_rad)])

    return np.linalg.norm(vec)


def calc_track_distance_traveled(track, still_displacement_threshold_px, still_time_threshold_frames):

    centroids = track['centroids'] #np.stack([np.array(ch.centroid) for ch in track['convex_hulls']])
    centroid_velocities = np.gradient(centroids, axis=-2)
    centroid_speeds = np.linalg.norm(centroid_velocities, axis=-1)

    centroid_still = np.ones(centroids.shape[0], dtype=np.bool)
    #for i in range(centroids.shape[0]):
    for i in range(0, centroids.shape[0], still_time_threshold_frames // 4):
        is_still = True
        for j in reversed(range(i + 1, min(centroids.shape[0], i + still_time_threshold_frames + 1))):

            if not centroid_still[j]:
                break

            if is_still:
                ij_displacement_vec = centroids[j, :] - centroids[i, :]
                ij_distance = np.linalg.norm(ij_displacement_vec)

                if ij_distance >= still_displacement_threshold_px:
                    is_still = False
                    centroid_still[i] = False
                    centroid_still[j] = False
            else:
                centroid_still[j] = False

    centroid_speeds[centroid_still] = 0

    return centroid_speeds.sum()


def _split_interval_by_displacement(pose_interval, track_relationship, maximum_displacement_px):
    interval_start, interval_stop = pose_interval
    interval_size = interval_stop - interval_start

    track1 = track_relationship['track1']
    track1_start_pose = track_relationship['track1_start_pose']
    track1_centroids = track1['centroids'][interval_start + track1_start_pose : interval_stop + track1_start_pose, :]

    track2 = track_relationship['track2']
    track2_start_pose = track_relationship['track2_start_pose']
    track2_centroids = track2['centroids'][interval_start + track2_start_pose : interval_stop + track2_start_pose, :]

    curr_track1_start_pos = track1_centroids[0, :]
    curr_track2_start_pos = track2_centroids[0, :]
    curr_start_offset = 0
    for interval_offset in range(1, interval_size):
        curr_track1_stop_pos = track1_centroids[interval_offset, :]
        curr_track2_stop_pos = track2_centroids[interval_offset, :]

        if (np.linalg.norm(curr_track1_stop_pos - curr_track1_start_pos) > maximum_displacement_px
            or np.linalg.norm(curr_track2_stop_pos - curr_track2_start_pos) > maximum_displacement_px):

            # we exceeded the distance so break up the interval here
            yield curr_start_offset + interval_start, interval_offset + interval_start

            # and reinitialize the start position data for the next interval
            curr_track1_start_pos = curr_track1_stop_pos
            curr_track2_start_pos = curr_track2_stop_pos
            curr_start_offset = interval_offset

    if curr_start_offset < interval_size - 1:
        yield curr_start_offset + interval_start, interval_stop


def detect_pairwise_proximity_intervals(
        track_relationship,
        maximum_distance_px,
        minimum_duration_frames,
        maximum_displacement_px,
        maximum_gap_merge_frames):

    proximity_candidate_frames = track_relationship['track_distances'] <= maximum_distance_px

    # find candidate intervals based on proximity
    proximity_intervals = find_intervals(proximity_candidate_frames, True)

    if maximum_displacement_px is not None:
        # break up candidate intervals where there is too much movement
        proximity_intervals = itertools.chain.from_iterable(
            _split_interval_by_displacement(hi, track_relationship, maximum_displacement_px)
            for hi in proximity_intervals
        )

    if minimum_duration_frames > 0:
        # remove intervals that are too short-lived
        proximity_intervals = (
            hi for hi in proximity_intervals
            if hi[1] - hi[0] >= minimum_duration_frames
        )

    if maximum_gap_merge_frames > 0:
        # merge proximity intervals that are close enough in time
        proximity_intervals = merge_intervals(proximity_intervals, maximum_gap_merge_frames)

    # move from track-relationship-relative, to absolute video frames
    relstart = track_relationship['start_frame']
    proximity_intervals = ((start + relstart, stop + relstart) for start, stop in proximity_intervals)

    return proximity_intervals


def _detect_approach_intervals(
        track_relationship,
        minimum_pre_approach_distance_px,
        maximum_arrival_distance_px,
        maximum_approach_duration_frames):

    pre_approach_candidate_frames = track_relationship['track_distances'] >= minimum_pre_approach_distance_px
    arrival_candidate_frames = track_relationship['track_distances'] <= maximum_arrival_distance_px

    for i in range(len(pre_approach_candidate_frames) - 1):

        if pre_approach_candidate_frames[i] and not pre_approach_candidate_frames[i + 1]:
            approach_start_frame = i + 1
            nearest_approach_distance = 0
            approach_stop_frame = -1

            for j in range(approach_start_frame, len(pre_approach_candidate_frames)):
                if pre_approach_candidate_frames[j] or j - approach_start_frame > maximum_approach_duration_frames:
                    break

                if arrival_candidate_frames[j]:
                    curr_distance = track_relationship['track_distances'][j]
                    assert curr_distance >= 0
                    if approach_stop_frame == -1 or curr_distance < nearest_approach_distance:
                        approach_stop_frame = j + 1
                        nearest_approach_distance = curr_distance

            if approach_stop_frame != -1:
                yield approach_start_frame, approach_stop_frame


def detect_approach_intervals(
        track_relationship,
        minimum_pre_approach_distance_px,
        maximum_arrival_distance_px,
        maximum_approach_duration_frames,
        maximum_still_speed_px_frame):

    # TODO it may be better to work off centroid speeds as is done in calc_track_distance_traveled

    intervals = _detect_approach_intervals(
        track_relationship,
        minimum_pre_approach_distance_px,
        maximum_arrival_distance_px,
        maximum_approach_duration_frames)

    for approach_start_frame, approach_stop_frame in intervals:

        track1 = track_relationship['track1']
        track1_approach_start_pose = approach_start_frame + track_relationship['track1_start_pose']
        track1_approach_stop_pose = approach_stop_frame + track_relationship['track1_start_pose']
        track1_max_speed = track1['mean_point_speeds'][track1_approach_start_pose:track1_approach_stop_pose].max()

        track2 = track_relationship['track2']
        track2_approach_start_pose = approach_start_frame + track_relationship['track2_start_pose']
        track2_approach_stop_pose = approach_stop_frame + track_relationship['track2_start_pose']
        track2_max_speed = track2['mean_point_speeds'][track2_approach_start_pose:track2_approach_stop_pose].max()

        # it's only considered an approach if one of the two mice are still
        if track1_max_speed < maximum_still_speed_px_frame or track2_max_speed < maximum_still_speed_px_frame:

            # the approaching animal is designated as the one with the higher maximum speed. We
            # call the approacher "track_a" and the mouse being approached "track_b"
            if track1_max_speed >= track2_max_speed:
                track_a_id = track1['track_id']
                track_a_centroid = track1['centroids'][track1_approach_start_pose, :] #np.array(track1['convex_hulls'][track1_approach_start_pose].centroid)

                track_b_id = track2['track_id']
                track_b_points = track2['points'][track2_approach_start_pose, ...]
                track_b_point_mask = track2['point_masks'][track2_approach_start_pose, ...]
            else:
                track_a_id = track2['track_id']
                track_a_centroid = track2['centroids'][track2_approach_start_pose, :] #np.array(track2['convex_hulls'][track2_approach_start_pose].centroid)

                track_b_id = track1['track_id']
                track_b_points = track1['points'][track1_approach_start_pose, ...]
                track_b_point_mask = track1['point_masks'][track1_approach_start_pose, ...]

            # We need to determine if it's an approach from front or back. To do this
            # we simply determine if the approaching animal is closer to the other animals
            # base of neck or base of tail.
            if track_b_point_mask[BASE_TAIL_INDEX] and track_b_point_mask[BASE_NECK_INDEX]:
                track_b_base_neck_xy = track_b_points[BASE_NECK_INDEX]
                track_b_base_tail_xy = track_b_points[BASE_TAIL_INDEX]

                dist_from_base_neck = np.linalg.norm(track_a_centroid - track_b_base_neck_xy)
                dist_from_base_tail = np.linalg.norm(track_a_centroid - track_b_base_tail_xy)

                if dist_from_base_tail < dist_from_base_neck:
                    approach_type = 'APPROACH_FROM_BEHIND'
                else:
                    approach_type = 'APPROACH_FROM_FRONT'

                yield {
                    'approacher_track_id': int(track_a_id),
                    'approached_track_id': int(track_b_id),
                    'approach_type': approach_type,
                    'start_frame': int(approach_start_frame + track_relationship['start_frame']),
                    'stop_frame_exclu': int(approach_stop_frame + track_relationship['start_frame']),
                }


def _calculate_gaze_angle_diff_deg(track_a_points, track_b_centroids):
    nose_xy = track_a_points[:, NOSE_INDEX].astype(np.double)
    base_neck_xy = track_a_points[:, BASE_NECK_INDEX].astype(np.double)

    nose_offset_xy = nose_xy - base_neck_xy
    gaze_angle_rad = np.arctan2(nose_offset_xy[:, 1], nose_offset_xy[:, 0])
    gaze_angle_deg = gaze_angle_rad * (180 / math.pi)

    track_offset_xy = track_b_centroids - base_neck_xy
    track_angle_rad = np.arctan2(track_offset_xy[:, 1], track_offset_xy[:, 0])
    track_angle_deg = track_angle_rad * (180 / math.pi)

    gaze_angle_deg[gaze_angle_deg < track_angle_deg] += 360
    gaze_angle_diff_deg = gaze_angle_deg - track_angle_deg
    gaze_angle_diff_deg[gaze_angle_diff_deg > 180] -= 360

    return gaze_angle_diff_deg


def _detect_watch_intervals(
        subject_data,
        object_data,
        maximum_gaze_offset_deg,
        within_dist_thresh_frames_arr,
        minimum_duration_frames,
        maximum_gap_merge_frames):

    s_track = subject_data['track']
    s_track_points = s_track['points'][subject_data['start_pose']:subject_data['stop_pose_exclu'], ...]
    s_track_point_masks = s_track['point_masks'][subject_data['start_pose']:subject_data['stop_pose_exclu'], ...]

    o_track = object_data['track']
    o_centroids = o_track['centroids'][object_data['start_pose']:object_data['stop_pose_exclu'], :]

    gaze_angle_diff_deg = _calculate_gaze_angle_diff_deg(s_track_points, o_centroids)
    gaze_offset_within_thresh = np.abs(gaze_angle_diff_deg) <= maximum_gaze_offset_deg

    watch_frames = (
        within_dist_thresh_frames_arr
        & gaze_offset_within_thresh
        & s_track_point_masks[:, NOSE_INDEX].astype(np.bool)
        & s_track_point_masks[:, BASE_NECK_INDEX].astype(np.bool)
    )

    watch_intervals = merge_intervals(
        find_intervals(watch_frames, True),
        maximum_gap_merge_frames,
    )
    watch_intervals = (
        (inter_start, inter_stop_exclu)
        for inter_start, inter_stop_exclu in watch_intervals
        if inter_stop_exclu - inter_start >= minimum_duration_frames
    )

    return watch_intervals


def detect_watch_intervals(
        track_relationship,
        maximum_gaze_offset_deg,
        minimum_distance_px,
        maximum_distance_px,
        minimum_duration_frames,
        maximum_gap_merge_frames):

    track1_data = {
        'track': track_relationship['track1'],
        'start_pose': track_relationship['track1_start_pose'],
        'stop_pose_exclu': track_relationship['track1_stop_pose_exclu'],
    }

    track2_data = {
        'track': track_relationship['track2'],
        'start_pose': track_relationship['track2_start_pose'],
        'stop_pose_exclu': track_relationship['track2_stop_pose_exclu'],
    }

    within_dist_thresh_frames_arr = np.logical_and(
        track_relationship['track_distances'] >= minimum_distance_px,
        track_relationship['track_distances'] <= maximum_distance_px,
    )

    track1_watching = _detect_watch_intervals(
        track1_data,
        track2_data,
        maximum_gaze_offset_deg,
        within_dist_thresh_frames_arr,
        minimum_duration_frames,
        maximum_gap_merge_frames)

    for inter_start, inter_stop_exclu in track1_watching:
        yield {
            'subject_track_id': int(track_relationship['track1']['track_id']),
            'object_track_id': int(track_relationship['track2']['track_id']),
            'start_frame': int(inter_start + track_relationship['start_frame']),
            'stop_frame_exclu': int(inter_stop_exclu + track_relationship['start_frame']),
        }

    track2_watching = _detect_watch_intervals(
        track2_data,
        track1_data,
        maximum_gaze_offset_deg,
        within_dist_thresh_frames_arr,
        minimum_duration_frames,
        maximum_gap_merge_frames)

    for inter_start, inter_stop_exclu in track2_watching:
        yield {
            'subject_track_id': int(track_relationship['track2']['track_id']),
            'object_track_id': int(track_relationship['track1']['track_id']),
            'start_frame': int(inter_start + track_relationship['start_frame']),
            'stop_frame_exclu': int(inter_stop_exclu + track_relationship['start_frame']),
        }
