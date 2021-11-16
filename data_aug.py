"""Augment data with three different techniques"""

import kaldiio
import numpy as np
from numpy import random
from numpy.core.defchararray import index
from numpy.lib.function_base import average
from scipy.stats import special_ortho_group as SO
from data_loading import build_segment_dicts, build_global_dvec_dict, build_meeting_dvec_dict

np.random.seed(0)

np.set_printoptions(threshold=np.inf)

def sub_meeting_augmentation(averaged_segmented_meetings_dict, segmented_speakers_dict,
                             meeting_length):
    """Sub-sequence randomisation.
       Randomly chooses a real meeting and samples a sub_meeting at segment boundaries.
       meeting_length is length of new meeting in number of segments.
    """
    # ensure chosen meeting is longer than required sub-sample length (meeting_length)
    valid_meeting_ids = np.array(list(averaged_segmented_meetings_dict.keys()))
    indexes_to_remove = []
    for i in range(len(valid_meeting_ids)):
        if len(averaged_segmented_meetings_dict[valid_meeting_ids[i]]) <= meeting_length:
            indexes_to_remove.append(i)
    valid_meeting_ids = np.delete(valid_meeting_ids, indexes_to_remove)
    if len(valid_meeting_ids) == 0:
        raise ValueError("meeting_length must be less than length of largest meeting in dataset")
    # randomly choose meeting
    random_meeting_id = np.random.choice(valid_meeting_ids)
    random_meeting = averaged_segmented_meetings_dict[random_meeting_id]
    # randomly choose starting index
    max_start_idx = len(random_meeting) - meeting_length
    random_start_idx = np.random.choice(max_start_idx+1)
    end_idx = random_start_idx + meeting_length - 1
    # produce sub-meeting
    augmented_meeting = random_meeting[random_start_idx:end_idx+1]
    augmented_speakers = segmented_speakers_dict[random_meeting_id][random_start_idx:end_idx+1]
    return augmented_meeting, augmented_speakers
    

# need to add shorten segments to less than 2s
def global_speaker_randomisation(global_dvec_dict, segmented_speakers_dict):
    """Global input vectors randomisation.
       Randomly sample a sequence of speaker labels.  For each label assign a speaker identity from
       any meeting.  For each segment in the sequence, sample a random d-vector from that speaker.
    """
    # each entry in array is list of speakers in a meeting
    speaker_labels_array = np.array(list(segmented_speakers_dict.values()), dtype=list)
    # choose random sequence of speaker labels
    random_speaker_seq = np.random.choice(speaker_labels_array)
    # create set of current unique speakers in sequence
    current_speakers = set(random_speaker_seq)
    # create set of all unique speakers available
    all_speakers = set(global_dvec_dict.keys())
    # create dictionary mapping current speakers to new speakers
    speaker_mapping = {}
    for current_speaker in current_speakers:
        new_speaker = np.random.choice(list(all_speakers))
        all_speakers.remove(new_speaker)  # prevents same new speaker being chosen twice
        speaker_mapping[current_speaker] = new_speaker
    # update speaker sequence with new speakers
    random_speaker_seq = [speaker_mapping[current_speaker] for current_speaker in random_speaker_seq]
    # create new meeting from label sequence, sampling random d-vectors from each speaker
    augmented_meeting = []
    for speaker in random_speaker_seq:
        random_idx = np.random.choice(len(global_dvec_dict[speaker]))
        random_dvec = global_dvec_dict[speaker][random_idx]
        augmented_meeting.append(random_dvec)
    return augmented_meeting, random_speaker_seq

# need to add shorten segments to less than 2s
def meeting_speaker_randomisation(meeting_dvec_dict, segmented_speakers_dict):
    """Meeting input vectors randomisation.
       Randomly sample a sequence of speaker labels.  Randomly choose a meeting with at least that
       number of speakers.  For each label assign a speaker identity from the chosen meeting.  For
       each segment in the sequence, sample a random d-vector from that speaker from that meeting.
    """
    # each entry in array is list of speakers in a meeting
    speaker_labels_array = np.array(list(segmented_speakers_dict.values()), dtype=list)
    # choose random sequence of speaker labels
    random_speaker_seq = np.random.choice(speaker_labels_array)
    # choose meeting to sample from, ensure it has at least the same number of speakers
    num_speakers = len(set(random_speaker_seq))
    valid_meeting_ids = np.array(list(meeting_dvec_dict.keys()))
    indexes_to_remove = []
    for i, meeting_id in enumerate(valid_meeting_ids):
        if len(set(segmented_speakers_dict[meeting_id])) < num_speakers:
            indexes_to_remove.append(i)
    np.delete(valid_meeting_ids, indexes_to_remove)
    random_meeting_id = np.random.choice(valid_meeting_ids)
    # create set of current unique speakers in sequence
    current_speakers = set(random_speaker_seq)
    # create set of unique speakers available from new meeting
    new_speakers = set(meeting_dvec_dict[random_meeting_id].keys())
    # create dictionary mapping current speakers to new speakers
    speaker_mapping = {}
    for current_speaker in current_speakers:
        new_speaker = np.random.choice(list(new_speakers))
        new_speakers.remove(new_speaker)  # prevents same new speaker being chosen twice
        speaker_mapping[current_speaker] = new_speaker
    # update speaker sequence with new speakers
    random_speaker_seq = [speaker_mapping[current_speaker] for current_speaker in random_speaker_seq]
    # create new meeting from label sequence, sampling random d-vectors from each speaker
    augmented_meeting = []
    for speaker in random_speaker_seq:
        random_idx = np.random.choice(len(meeting_dvec_dict[meeting_id][speaker]))
        random_dvec = meeting_dvec_dict[meeting_id][speaker][random_idx]
        augmented_meeting.append(random_dvec)
    return augmented_meeting, random_speaker_seq

def Diaconis(batch):
    """Randomly samples a rotation matrix and applies to each meeting in a batch of data.
       Returns the rotated batch.
    """
    dimension = 32  # of d-vector
    rotation_mat = SO.rvs(dimension)
    for meeting_id in batch:
        batch[meeting_id] = np.array(batch[meeting_id])
        # rotate meeting
        batch[meeting_id] = np.dot(batch[meeting_id], rotation_mat)
        # normalise variance
        batch[meeting_id] *= np.sqrt(dimension)
    return batch


def produce_augmented_batch(averaged_segmented_meetings_dict=None, segmented_speakers_dict=None,
                            global_dvec_dict=None, meeting_dvec_dict = None, batch_size=10,
                            aug_type="global"):
    """Produces a batch of augmented data for training.
       The dicts contain original meetings.  Only dicts corresponding to aug_types are required.
       aug_type is a string which can be either "global" or "meeting".
       batch_size is number of new meetings to be produced
    """
    # Two dictionaries with key as new meeting_id
    aug_meetings = {}  # Value is augmented meeting (1 d-vector per segment)
    aug_speakers = {}  # Value is labels for meeting (1 speaker per segment)

    # first do sub-meeting
    for i in range(batch_size//2):
        aug_meeting_id = "AUG_" + str(i)
        # randomly choose meeting length
        meeting_length = np.random.choice(np.arange(100, 1000))
        aug_meeting, aug_speaker = sub_meeting_augmentation(averaged_segmented_meetings_dict,
                                                        segmented_speakers_dict, meeting_length)
        aug_meetings[aug_meeting_id] = aug_meeting
        aug_speakers[aug_meeting_id] = aug_speaker
        
    if aug_type == "global":
        for i in range(batch_size//2, batch_size):
            aug_meeting, aug_speaker = global_speaker_randomisation(global_dvec_dict,
                                                                segmented_speakers_dict)
    elif aug_type == "meeting":
        for i in range(batch_size//2, batch_size):
            aug_meeting, aug_speaker = meeting_speaker_randomisation(meeting_dvec_dict,
                                                                    segmented_speakers_dict)

    # do Diac aug on entire batch
    diac_aug_meetings = Diaconis(aug_meetings)

    return diac_aug_meetings, aug_speakers


def main():
    """Main"""
    dataset = "dev"

    # averaged_segmented_meetings_dict, segmented_speakers_dict = build_segment_dicts(dataset)
    # aug_meeting, aug_speakers = sub_meeting_augmentation(averaged_segmented_meetings_dict, segmented_speakers_dict, 300)

    # global_dvec_dict = build_global_dvec_dict(dataset)
    # _, segmented_speakers_dict = build_segment_dicts(dataset)
    # aug_meeting, aug_speakers = global_speaker_randomisation(global_dvec_dict, segmented_speakers_dict)

    # meeting_dvec_dict = build_meeting_dvec_dict(dataset)
    # _, segmented_speakers_dict = build_segment_dicts(dataset)
    # aug_meeting, aug_speakers = meeting_speaker_randomisation(meeting_dvec_dict, segmented_speakers_dict)

    averaged_segmented_meetings_dict, segmented_speakers_dict = build_segment_dicts(dataset)
    global_dvec_dict = build_global_dvec_dict(dataset)
    aug_meetings, aug_speakers = produce_augmented_batch(
                                 averaged_segmented_meetings_dict=averaged_segmented_meetings_dict,
                                 segmented_speakers_dict=segmented_speakers_dict,
                                 global_dvec_dict=global_dvec_dict,
                                 batch_size=5,
                                 aug_type="global")
    print(aug_meetings, aug_speakers)


if __name__ == '__main__':
    main()
    