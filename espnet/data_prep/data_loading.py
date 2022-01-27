"""Loading meeting and segmentation data from scp and rttm files."""

import kaldiio
import numpy as np
from collections import defaultdict


def open_rttm(rttm_path):
    """Open rttm file containing segmentation data.

    :param: str rttm_path: path to rttm file
    :return: List[List[str]] segments_desc_list: list of each line of the file as a lists of strings
    """
    with open(rttm_path, "r") as rttm_file:
        segments_desc_list = [(line.strip()).split() for line in rttm_file]
    return segments_desc_list


def build_segment_desc_dict(rttm_path):
    """Build dictionary segment_desc_dict.

    :param: str rttm_path: path to rttm file
    :return: dict segment_desc_dict[meeting_id] =  np.array(start_index, end_index, speaker_label,
    start_time, end_time, duration)
    """
    segment_desc_dict = defaultdict(list)
    segments_desc_list = open_rttm(rttm_path)
    for segment_desc in segments_desc_list:
        meeting_id = segment_desc[1]
        start_index = int(segment_desc[5])
        end_index = int(segment_desc[6])
        speaker_label = segment_desc[7]
        start_time = float(segment_desc[3])
        duration = float(segment_desc[4])
        end_time = start_time + duration
        segment_desc_dict[meeting_id].append((start_index, end_index, speaker_label,
                                              start_time, end_time, duration))
    for meeting_id, segment_descs in segment_desc_dict.items():  # filter encompassed segments
        segment_desc_dict[meeting_id] = filter_encompassed_segments(segment_descs)
    return segment_desc_dict


def open_scp(scp_path):
    """Open scp file containing paths to meeting d-vectors and return numpy array.

    :param: str scp_path: path to scp file
    :return: List[List[str]] meeting_path_lists: List of Lists [meeting_id, path_to_ark_file]
    """
    with open(scp_path, "r") as scp_file:
        meeting_path_lists = [(line.strip()).split() for line in scp_file]
    return meeting_path_lists


def build_global_dvec_dict(dataset, split=False):
    """Builds global d-vector dictionary (d-vectors from across all meetings)
    
    :param: str dataset: "train", "dev", or "eval"
    :param: Bool split: splits segments longer than 2s if True
    :return: dict global_dvec_dict[speaker_label] = List[dvector] where dvector is 32-D np array
    """

    scp_path, rttm_path = get_file_paths(dataset)
    global_dvec_dict = {}
    meeting_path_lists = open_scp(scp_path)
    segment_desc_dict = build_segment_desc_dict(rttm_path)
    for meeting_path_list in meeting_path_lists:  # iterate through meetings
        meeting_id = meeting_path_list[0]
        meeting_path = meeting_path_list[1]
        meeting_dvectors_array = kaldiio.load_mat(meeting_path)
        for segment_desc in segment_desc_dict[meeting_id]:
            start_index = segment_desc[0]
            end_index = segment_desc[1]
            segment = meeting_dvectors_array[start_index:end_index]
            speaker = segment_desc[2]
            if split:
                # split segments longer than 2s to give more training examples
                num_subsegments = max(1, len(segment) // 100)
                subsegments = np.array_split(segment, num_subsegments)
            else:
                subsegments = [segment]
            for subsegment in subsegments:
                averaged_subsegment = np.mean(subsegment, axis=0)
                averaged_subsegment = averaged_subsegment/np.linalg.norm(averaged_subsegment)
                if speaker not in global_dvec_dict:
                    global_dvec_dict[speaker] = [averaged_subsegment]
                else:
                    global_dvec_dict[speaker].append(averaged_subsegment)

    return global_dvec_dict


def build_meeting_dvec_dict(dataset, split=False):
    """Build meeting-level d-vector dictionary (dictionary of dictionaries, one per meeting).
    
    :param: str dataset: "train", "dev", or "eval"
    :param: Bool split: splits segments longer than 2s if True
    :return: dict meeting_dvec_dict[meeting_id] = {speaker_label: List[dvector]}
    """
    scp_path, rttm_path = get_file_paths(dataset)
    meeting_dvec_dict = {}
    meeting_path_lists = open_scp(scp_path)
    segment_desc_dict = build_segment_desc_dict(rttm_path)
    for meeting_path_list in meeting_path_lists:  # iterate through meetings
        inner_dvec_dict = {}  # to be a value in meeting_dvec_dict (defaultdict for numpy array?)
        meeting_id = meeting_path_list[0]
        meeting_path = meeting_path_list[1]
        meeting_dvectors_array = kaldiio.load_mat(meeting_path)
        for segment_desc in segment_desc_dict[meeting_id]:
            start_index = segment_desc[0]
            end_index = segment_desc[1]
            segment = meeting_dvectors_array[start_index:end_index]
            speaker = segment_desc[2]
            if split:
                # split segments longer than 2s to give more training examples
                num_subsegments = max(1, len(segment) // 100)
                subsegments = np.array_split(segment, num_subsegments)
            else:
                subsegments = [segment]
            for subsegment in subsegments:
                averaged_subsegment = np.mean(subsegment, axis=0)
                averaged_subsegment = averaged_subsegment/np.linalg.norm(averaged_subsegment)
                if speaker not in inner_dvec_dict:
                    inner_dvec_dict[speaker] = [averaged_subsegment]
                else:
                    inner_dvec_dict[speaker].append(averaged_subsegment)
        meeting_dvec_dict[meeting_id] = inner_dvec_dict
    return meeting_dvec_dict


def build_segment_dicts(dataset):
    """Build averaged_segmented_meetings_dict and segmented_speakers_dict (labels).

    :param: str dataset: "train", "dev", or "eval"
    :return: dict averaged_segmented_meetings_dict[meeting_id] = List[dvector] (Sequence of segments
            for each meeting)
    :return: dict segmented_speakers_dict[meeting_id] = List[str] (Sequence of speaker labels for each
            meeting)
    """
    scp_path, rttm_path = get_file_paths(dataset)
    # create two dictionaries with key as meeting_id:
    segmented_speakers_dict = {}  # value is array of speakers aligning with segments
    averaged_segmented_meetings_dict = {}  # value is array of segments.  Each segment is 1 d-vector
    meeting_path_lists = open_scp(scp_path)
    segment_desc_dict = build_segment_desc_dict(rttm_path)
    for meeting_path_list in meeting_path_lists:  # iterate through meetings
        meeting_id = meeting_path_list[0]
        meeting_path = meeting_path_list[1]
        meeting_dvectors_array = kaldiio.load_mat(meeting_path)
        speakers = []
        averaged_segments = []
        for segment_desc in segment_desc_dict[meeting_id]:
            start_index = segment_desc[0]
            end_index = segment_desc[1]
            segment = meeting_dvectors_array[start_index:end_index]
            averaged_segment = np.mean(segment, axis=0)
            averaged_segment = averaged_segment/np.linalg.norm(averaged_segment)
            speaker = segment_desc[2]
            speakers.append(speaker)
            averaged_segments.append(averaged_segment)
        averaged_segmented_meetings_dict[meeting_id] = averaged_segments
        segmented_speakers_dict[meeting_id] = speakers
    return averaged_segmented_meetings_dict, segmented_speakers_dict


def get_file_paths(dataset):
    """Get path for chosen dataset.
    Dataset can be either 'train', 'dev' or 'eval'.
    """
    if dataset == 'train':
        scp_path = "/home/mifs/jhrt2/newDNC/data/arks.concat/train.scp"
        rttm_path = "/home/mifs/jhrt2/newDNC/data/rttms.concat/train.rttm"
    elif dataset == 'dev':
        scp_path = "/home/mifs/jhrt2/newDNC/data/arks.concat/dev.scp"
        rttm_path = "/home/mifs/jhrt2/newDNC/data/rttms.concat/dev.rttm"
    elif dataset == 'eval':
        scp_path = "/home/mifs/jhrt2/newDNC/data/arks.concat/eval.scp"
        rttm_path = "/home/mifs/jhrt2/newDNC/data/rttms.concat/eval.rttm"
    else:
        raise ValueError("Expected dataset argument to be 'train', 'dev' or 'eval'")
    return scp_path, rttm_path


def filter_encompassed_segments(_seg_list):
    """Remove segments completely contained within another one based on time (not indices).
    Takes segment_desc_list from build_segment_desc_dict()

    :param: _seg_list np.array(segment_information)
    :return: seg_list np.array(segment_information)
    """
    _seg_list.sort(key=lambda tup: tup[3])
    seg_list = []
    for segment in _seg_list:
        start_time = segment[3]
        end_time = segment[4]
        start_before = [_seg for _seg in _seg_list if _seg[3] <= start_time]
        end_after = [_seg for _seg in _seg_list if _seg[4] >= end_time]
        start_before.remove(segment)
        end_after.remove(segment)
        if set(start_before).isdisjoint(end_after):
            seg_list.append(segment)
    return seg_list
