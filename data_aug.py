"""Augment data with three different techniques"""

import kaldiio
import numpy as np
from numpy import random
from numpy.core.defchararray import index
from numpy.lib.function_base import average
from data_loading import build_segment_dicts, build_global_dvec_dict

np.random.seed(0)


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
    """Input vectors randomisation.
       Randomly sample sequence of speaker labels.  For each label assign a speaker identity.
       For each segment in the sequence, sample a random d-vector from that speaker.
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


def meeting_speaker_randomisation():
    pass


def Diaconis_augmentation():
    pass


def produce_augmented_batch(batch_size, sub=True, speaker=True, Diac=True):
    """Produces a batch of augmented data for training.
       batch size and augmentation types can be specified.
    """
    pass


def main():
    """Main"""
    dataset = "dev"
    averaged_segmented_meetings_dict, segmented_speakers_dict = build_segment_dicts(dataset)
    aug_meeting, aug_speakers = sub_meeting_augmentation(averaged_segmented_meetings_dict, segmented_speakers_dict, 300)

    # global_dvec_dict = build_global_dvec_dict(dataset)
    # _, segmented_speakers_dict = build_segment_dicts(dataset)
    # #print([len(meeting) for meeting in segmented_speakers_dict.values()])
    # aug_meeting = global_speaker_randomisation(global_dvec_dict, segmented_speakers_dict)


if __name__ == '__main__':
    main()
    