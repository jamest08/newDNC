"""Loading meeting and segmentation data from scp and rttm files."""

import kaldiio
import numpy as np
import pickle


def open_rttm(rttm_path):
    """Open rttm file containing segmentation data and return as numpy array."""
    rttm_file = open(rttm_path, "r")
    segments_desc_list = [(line.strip()).split() for line in rttm_file]
    rttm_file.close()
    return segments_desc_list


def build_segment_desc_dict(rttm_path):
    """Build dictionary segment_desc_dict.

    Key is meeting name, value is numpy array of tuples (start_time, duration, speaker_label)
    """
    segment_desc_dict = {}
    segments_desc_list = open_rttm(rttm_path)
    for segment_desc in segments_desc_list:
        meeting_id = segment_desc[1]
        start_point = float(segment_desc[3])
        duration = float(segment_desc[4])
        speaker_label = segment_desc[7]
        if meeting_id not in segment_desc_dict:
            segment_desc_dict[meeting_id] = [(start_point, duration, speaker_label)]
        else:
            segment_desc_dict[meeting_id].append((start_point, duration, speaker_label))
    return segment_desc_dict


def open_scp(scp_path):
    """Open scp file containing paths to meeting d-vectors and return numpy array."""
    scp_file = open(scp_path, "r")
    # create list of lists [meeting_id, path_to_ark]
    meeting_path_lists = [(line.strip()).split() for line in scp_file]
    scp_file.close()
    return meeting_path_lists


def build_dvec_dict(dataset):
    """Create global dvec_dict and two supoorting dictionaries for testing.

    Dataset can be either 'train', 'dev' or 'eval'
    """
    if dataset == 'train':
        scp_path = "data/arks/train.scp"
        rttm_path = "data/rttms/test_train.rttm"
    elif dataset == 'dev':
        scp_path = "data/arks/dev.scp"
        rttm_path = "data/rttms/test_dev.rttm"
    elif dataset == 'eval':
        scp_path = "data/arks/dev.scp"
        rttm_path = "data/rttms/test_dev.rttm"
    else:
        raise ValueError("Expected dataset argument to be 'train', 'dev' or 'eval'")
    # create three dictionaries each with key as meeting_id:
    segmented_meetings_dict = {}  # value is array of segments.  Each segment is array of d-vectors
    segmented_speakers_dict = {}  # value is array of speakers aligning with segments
    averaged_segmented_meetings_dict = {}  # value is array of segments.  Each segment is 1 d-vector
    # simultaneously build d-vector dictionary
    # key is speaker, value is list of all d_vectors for that speaker (across all meetings)
    global_dvec_dict = {}
    # key is meeting id, value is another dictionary
    # where key is speaker and value is list of dvectors for that speaker and meeting
    meeting_dvec_dict = {}
    meeting_path_lists = open_scp(scp_path)
    segment_desc_dict = build_segment_desc_dict(rttm_path)
    window_shift_length = 0.01  # d-vectors cover a 2s period and start every 10ms (overlapping)
    for meeting_path_list in meeting_path_lists:  # iterate through meetings
        inner_dvec_dict = {}  # to be a value in meeting_dvec_dict
        meeting_id = meeting_path_list[0]
        meeting_path = meeting_path_list[1]
        meeting_dvectors_array = kaldiio.load_mat(meeting_path)
        segments = []
        speakers = []
        averaged_segments = []
        for segment_desc in segment_desc_dict[meeting_id]:
            start_time = segment_desc[0]
            end_time = segment_desc[0] + segment_desc[1]  # start + duration
            start_index = max(int(round(start_time / window_shift_length)), 0)
            end_index = min(int(round(end_time / window_shift_length)), len(meeting_dvectors_array))
            segment = meeting_dvectors_array[start_index:end_index]
            averaged_segment = np.mean(segment, axis=0)
            speaker = segment_desc[2]
            segments.append(segment)
            speakers.append(speaker)
            averaged_segments.append(averaged_segment)
            if speaker not in global_dvec_dict:
                global_dvec_dict[speaker] = segment
            else:
                global_dvec_dict[speaker] = np.append(global_dvec_dict[speaker], segment, axis=0)
            if speaker not in inner_dvec_dict:
                inner_dvec_dict[speaker] = segment
            else:
                inner_dvec_dict[speaker] = np.append(inner_dvec_dict[speaker], segment, axis=0)
        segmented_meetings_dict[meeting_id] = segments  #should these be numpy arrays
        segmented_speakers_dict[meeting_id] = speakers
        averaged_segmented_meetings_dict[meeting_id] = averaged_segments
        meeting_dvec_dict[meeting_id] = inner_dvec_dict
    return global_dvec_dict, meeting_dvec_dict, segmented_meetings_dict, \
           segmented_speakers_dict, averaged_segmented_meetings_dict

def save_obj(obj, name):
    """Saves an object to /obj using pickle."""
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# only build_dvec_dict should be called directly
print(kaldiio.load_mat("/home/dawna/flk24/diarization/NeuralSpeakerClustering/extractDVector.Paper.Meeting/data/arks/train.ark:16"))
global_dvec_dict, meeting_dvec_dict, segmented_meetings_dict, segmented_speakers_dict, averaged_segmented_meetings_dict = build_dvec_dict("train")
# print(global_dvec_dict["FIE038"])
# print(meeting_dvec_dict["AMIMDM-00IB4005"]["FIE038"])
# print(segmented_meetings_dict["AMIMDM-00IB4005"])  # for testing
print(segmented_speakers_dict["AMIMDM-00IB4005"])
print(averaged_segmented_meetings_dict["AMIMDM-00IB4005"])



save_obj(global_dvec_dict, "global_dvec_dict")
save_obj(meeting_dvec_dict, "meeting_dvec_dict")
save_obj(segmented_meetings_dict, "segmented_meetings_dict")
save_obj(segmented_speakers_dict, "segmented_speakers_dict")
save_obj(averaged_segmented_meetings_dict, "averaged_segmented_meetings_dict")

assert(len(averaged_segmented_meetings_dict["AMIMDM-00IB4005"]) == len(segmented_speakers_dict["AMIMDM-00IB4005"]))
assert(len(averaged_segmented_meetings_dict["AMIMDM-00IB4005"]) == len(segmented_meetings_dict["AMIMDM-00IB4005"]))
