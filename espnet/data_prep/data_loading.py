"""Loading meeting, segmentation and TDOA data from scp, rttm and del files."""

from functools import partial
from typing import List
import kaldiio
import numpy as np
from collections import defaultdict
import os

from torch import float32
import configargparse

np.set_printoptions(threshold=np.inf)

def open_rttm(rttm_path):
    """Open rttm file containing segmentation data.

    :param: str rttm_path: path to rttm file
    :return: List[List[str]] segments_desc_list: list of each line of the file as a lists of strings
    """
    with open(rttm_path, "r") as rttm_file:
        segments_desc_list = [(line.strip()).split() for line in rttm_file]
    return segments_desc_list


def build_segment_desc_dict(rttm_path, filt=True):
    """Build dictionary segment_desc_dict.

    :param: str rttm_path: path to rttm file
    :return: dict segment_desc_dict[meeting_id] =  List(Tuple(start_index, end_index, speaker_label,
    start_time, end_time, duration))
    """
    segment_desc_dict = defaultdict(list)
    segments_desc_list = open_rttm(rttm_path)
    for segment_desc in segments_desc_list:
        meeting_id = segment_desc[1]
        try:
            start_index = int(segment_desc[5])
        except ValueError:  # if '<NA>'
            start_index = segment_desc[5]
        try:
            end_index = int(segment_desc[6])
        except ValueError:  # if '<NA>'
            end_index = segment_desc[6]
        speaker_label = segment_desc[7]
        start_time = round(float(segment_desc[3]), 2)
        duration = round(float(segment_desc[4]), 2)
        end_time = round(start_time + duration, 2)
        segment_desc_dict[meeting_id].append((start_index, end_index, speaker_label,
                                              start_time, end_time, duration))
    if filt:
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


def build_global_dvec_dict(args, dataset, split=False, tdoa=False, gccphat=False):
    """Builds global d-vector dictionary (d-vectors from across all meetings)
    
    :param: str dataset: "train", "dev", or "eval"
    :param: Bool split: splits segments longer than 2s if True
    :return: dict global_dvec_dict[speaker_label] = List[dvector] where dvector is 32-D np array
    """

    scp_path, rttm_path = get_file_paths(args, dataset)
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


def build_meeting_dvec_dict(args, dataset, split=False, tdoa=False, gccphat=False):
    """Build meeting-level d-vector dictionary (dictionary of dictionaries, one per meeting).
    
    :param: str dataset: "train", "dev", or "eval"
    :param: Bool split: splits segments longer than 2s if True
    :return: dict meeting_dvec_dict[meeting_id] = {speaker_label: List[dvector]}
    """
    scp_path, rttm_path = get_file_paths(args, dataset)
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


def build_segment_dicts(args, dataset, filt=True, dvec=True, tdoa=False, gccphat=False, average=True):
    """Build averaged_segmented_meetings_dict and segmented_speakers_dict (labels).

    :param: str dataset: "train", "dev", or "eval"
    :param: Bool filt: apply filtered_encomassed_segments
    :param: Bool dvec: include d-vectors
    :param: Bool tdoa: include TDOA values
    :param: Bool gccphat: include GCC-PHAT values
    :param: Bool average: average segments or leave as array
    :return: dict averaged_segmented_meetings_dict[meeting_id] = List[dvector] (Sequence of segments
            for each meeting.  Each vector has some combination of dvec, tdoa, gccphat in that order)
    :return: dict segmented_speakers_dict[meeting_id] = List[str] (Sequence of speaker labels for each
            meeting)
    """
    print("Building segment dicts", 'dvec: ' + str(dvec), 'tdoa: ' + str(tdoa), 'gccphat: ' + str(gccphat), 'average: ' + str(average))
    scp_path, rttm_path = get_file_paths(args, dataset)
    # create two dictionaries with key as meeting_id:
    segmented_speakers_dict = {}  # value is array of speakers aligning with segments
    segmented_meetings_dict = {}  # value is array of segments.  Each segment is 1 d-vector
    meeting_path_lists = open_scp(scp_path)
    segment_desc_dict = build_segment_desc_dict(rttm_path, filt=filt)

    if tdoa == True or gccphat == True:
        tdoas, gccphats = get_tdoa_gccphat(args, segment_desc_dict.keys())

    for meeting_path_list in meeting_path_lists:  # iterate through meetings
        meeting_id = meeting_path_list[0]
        meeting_path = meeting_path_list[1]
        if dvec == True:
            meeting_dvectors = kaldiio.load_mat(meeting_path)
            dvec_dim =  meeting_dvectors.shape[1]  # 32
            meeting = meeting_dvectors

        # concatenate arrays, ignore final repeated/padding TDOA and GCC-PHAT values
        if tdoa == True:
            if dvec == True:
                meeting_tdoas = tdoas[meeting_id][:len(meeting)]
                meeting = np.concatenate((meeting, meeting_tdoas), axis=1, dtype=np.float32)
            else:
                meeting = np.array(tdoas[meeting_id][:-12], dtype=np.float32)
        if gccphat == True:
            if dvec == True or tdoa == True:
                meeting_gccphats = gccphats[meeting_id][:len(meeting)]
                meeting = np.concatenate((meeting, meeting_gccphats), axis=1, dtype=np.float32)
            else:
                meeting = np.array(gccphats[meeting_id][:-12], dtype=np.float32)

        speakers = []
        segments = []
        for segment_desc in segment_desc_dict[meeting_id]:
            start_index = segment_desc[0]
            end_index = segment_desc[1]
            segment = meeting[start_index:end_index]
            # take average regardless of data included
            if average == True:
                segment = np.mean(segment, axis=0)
            # only normalise dvec part
            if dvec == True:
                segment[:dvec_dim] = segment[:dvec_dim]/np.linalg.norm(segment[:dvec_dim])
            speaker = segment_desc[2]
            speakers.append(speaker)
            segments.append(segment)
        segmented_meetings_dict[meeting_id] = segments
        segmented_speakers_dict[meeting_id] = speakers
    return segmented_meetings_dict, segmented_speakers_dict


def get_file_paths(args, dataset):
    """Get path for chosen dataset.
    Dataset can be either 'train' or 'dev'.
    """
    if dataset == 'train':
        scp_path = args.train_scp
        rttm_path = args.train_rttm
    elif dataset == 'dev':
        scp_path = args.valid_scp
        rttm_path = args.valid_rttm
    elif dataset == 'eval':
        scp_path = args.eval_scp
        rttm_path = args.eval_rttm
    else:
        raise ValueError("Expected dataset argument to be 'train', 'dev' or 'eval")
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


def get_tdoa_gccphat(args, meeting_ids, norm=True):
    """Returns two dicts storing TDOA and GCC-PHAT values for meetings in a dataset.
    
    :param: str args.directory_path: path to directory containing del files
    :param: List[str] meeting_ids: list of meeting_ids to get data for
    :param: Bool norm: if True, normalise values at corpus level for have zero mean and unit variance

    :return: dict tdoas[meeting_id] = List[np.array(int)] Each entry in list is a vector
                                                  of TDOA values aligned with d-vectors.
    :return: dict gccphats[meeting_id] = List[np.array(float)] Each entry in list is a vector
                                                  of GCC-PHAT values aligned with d-vectors.                                            
    """
    print('Getting TDOA/GCCPHAT data')
    directory_path = args.tdoa_directory

    tdoas = defaultdict(list)
    gccphats = defaultdict(list)

    # all_tdoa_vecs = []
    # all_gccphat_vecs = []

    for meeting_id in meeting_ids:
        partial_meeting_id = meeting_id[8:]
        with open(directory_path + '/' + partial_meeting_id + '.del', 'r') as delays:
            delays_list = [(line.strip()).split() for line in delays]
        for delay in delays_list:
            # ignore second channel as that is fixed reference (always 0 1.000000)
            tdoa_vec = np.array([int(delay[2]), int(delay[6]), int(delay[8]), int(delay[10]),
                                int(delay[12]), int(delay[14]), int(delay[16])])
            gccphat_vec = np.array([float(delay[3]), float(delay[7]), float(delay[9]), float(delay[11]),
                                float(delay[13]), float(delay[15]), float(delay[17])])
            tdoas[meeting_id].append(tdoa_vec)
            gccphats[meeting_id].append(gccphat_vec)
            # all_tdoa_vecs.append(tdoa_vec)
            # all_gccphat_vecs.append(gccphat_vec)

        tdoas[meeting_id] = np.array(tdoas[meeting_id], dtype=np.float32)
        gccphats[meeting_id] = np.array(gccphats[meeting_id], dtype=np.float32)

    if norm == True:
        # all_tdoa_vecs = np.array(all_tdoa_vecs, dtype=np.float32)
        # all_gccphat_vecs = np.array(all_gccphat_vecs, dtype=np.float32)
        # tdoa_mean = np.mean(all_tdoa_vecs, axis=0)  # mean vector 
        # gccphat_mean = np.mean(all_gccphat_vecs, axis=0)
        # tdoa_std = np.std(all_tdoa_vecs, axis=0)  # standard deviation vector
        # gccphat_std = np.std(all_gccphat_vecs, axis=0)

        # hard code stats from get_tdoa_gccphat_stats to save time (from train set)
        tdoa_mean = np.array([0.20190075, -0.05194093, 0.14035095, 0.52727616, 0.8346257, 1.0925584, 0.83129424], dtype=np.float32)
        tdoa_std = np.array([2.0948744, 2.3241794, 5.6177387, 5.9525986, 6.925673, 5.7009034, 4.0400524], dtype=np.float32)
        gccphat_mean = np.array([0.27077588, 0.26317957, 0.24773844, 0.23016952, 0.22052345, 0.21291934, 0.23988448], dtype=np.float32)
        gccphat_std = np.array([0.14447168, 0.13321947, 0.12776719, 0.12591319, 0.13311823, 0.12614751, 0.1267838] , dtype=np.float32)
        for meeting_id in tdoas:
            # normalise so data has zero mean and unit variance
            tdoas[meeting_id] = np.divide((tdoas[meeting_id] - tdoa_mean), tdoa_std)
            gccphats[meeting_id] = np.divide((gccphats[meeting_id] - gccphat_mean), gccphat_std)

    return tdoas, gccphats


def get_parser():  # debugging only, official paths should be maintained in asr_train.py
    parser = configargparse.ArgumentParser(
        description="Load speech data",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train-scp', type=str,
            default="/home/mifs/jhrt2/newDNC/data/arks.meeting.cmn.tdnn/train.scp", help='')
    parser.add_argument('--valid-scp', type=str,
            default="/home/mifs/jhrt2/newDNC/data/arks.meeting.cmn.tdnn/dev.scp", help='')
    parser.add_argument('--train-rttm', type=str,
            default="/home/mifs/jhrt2/newDNC/espnet/data_prep/train_window_level.rttm", help='')
    parser.add_argument('--valid-rttm', type=str,
            default="/home/mifs/jhrt2/newDNC/espnet/data_prep/dev_window_level.rttm", help='')
    parser.add_argument('--tdoa-directory', type=str,
            default="/data/mifs_scratch/jhrt2/BeamformIt/MDM_AMI_fixedref_10", help='')
    return parser

def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    dataset = 'train'

    meetings, speakers = build_segment_dicts(args, dataset, filt=True, dvec=True, tdoa=True, gccphat=True, average=True)

if __name__ == '__main__':
    main()

