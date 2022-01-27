"""Augment data with three different techniques"""

import json
import os
import kaldiio
import numpy as np
from scipy.stats import special_ortho_group as SO

try:  # use this one for generator
    from data_prep.data_loading import build_segment_dicts, build_global_dvec_dict, build_meeting_dvec_dict
except:  # use this for debugging
    from data_loading import build_segment_dicts, build_global_dvec_dict, build_meeting_dvec_dict


def sub_meeting_augmentation(averaged_segmented_meetings_dict, segmented_speakers_dict,
                             meeting_length=50):
    """Sub-sequence randomisation.
       Randomly chooses a real meeting and samples a sub_meeting at segment boundaries.

       :param: dict averaged_segmented_meetings_dict[meeting_id] = List[dvector] (Sequence of segments
            for each meeting)
       :param: dict segmented_speakers_dict[meeting_id] = List[str] (Sequence of speaker labels for
            each meeting)
       :param: int meeting_length: desired length of augmented meeting in number of segments
       :return: List[dvectors] augmented_meeting: the new, augmented meeting (dvector sequence)
       :return: List[str] augmented_speaker: the new, augmented sequence of speaker labels
    """
    # ensure chosen meeting is longer than required sub-sample length (meeting_length)
    # TODO: this doesn't need to be repeated on each call.  Not urgent as not actually too slow
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
    # randomly choose starting index (ie. starting segment)
    max_start_idx = len(random_meeting) - meeting_length
    random_start_idx = np.random.choice(max_start_idx+1)
    end_idx = random_start_idx + meeting_length - 1
    # produce sub-meeting
    augmented_meeting = random_meeting[random_start_idx:end_idx+1]
    augmented_speakers = segmented_speakers_dict[random_meeting_id][random_start_idx:end_idx+1]
    return augmented_meeting, augmented_speakers


def global_speaker_randomisation(global_dvec_dict, segmented_speakers_dict, meeting_length=50):
    """Global input vectors randomisation.
        Randomly sample a sequence of speaker labels.  For each label assign a speaker identity from
        any meeting.  For each segment in the sequence, sample a random d-vector from that speaker.

        :param: dict global_dvec_dict[speaker_label] = List[dvector] where dvector is 32-D np array
        :param: dict segmented_speakers_dict[meeting_id] = List[str] (Sequence of speaker labels for
            each meeting)
        :param: int meeting_length: desired length of augmented meeting in number of segments
        :return: List[dvectors] augmented_meeting: the new, augmented meeting (dvector sequence)
        :return: List[str] random_speaker_seq: the new, augmented sequence of speaker labels
    """
    # each entry in array is list of speakers in a meeting
    speaker_labels_array = np.array(list(segmented_speakers_dict.values()), dtype=list)
    # choose random sequence of speaker labels
    random_speaker_seq = np.random.choice(speaker_labels_array)
    # randomly truncate sequence to meeting length (effectively sub-meeting randomisation)
    if meeting_length < len(random_speaker_seq):
        start_index = np.random.choice(len(random_speaker_seq) - meeting_length + 1)
        end_index = start_index + meeting_length - 1
        random_speaker_seq = random_speaker_seq[start_index:end_index+1]
    else:  # TODO: this needs to be improved. just for testing.  shortest train meeting is 71
        print("shorter meeting returned")
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


def meeting_speaker_randomisation(meeting_dvec_dict, segmented_speakers_dict, meeting_length):
    """Meeting input vectors randomisation.
        Randomly sample a sequence of speaker labels.  Randomly choose a meeting with at least that
        number of speakers.  For each label assign a speaker identity from the chosen meeting.  For
        each segment in the sequence, sample a random d-vector from that speaker from that meeting.

        :param: dict meeting_dvec_dict[meeting_id] = {speaker_label: List[dvector]}
        :param: dict segmented_speakers_dict[meeting_id] = List[str] (Sequence of speaker labels for
            each meeting)
        :param: int meeting_length: desired length of augmented meeting in number of segments
        :return: List[dvectors] augmented_meeting: the new, augmented meeting (dvector sequence)
        :return: List[str] random_speaker_seq: the new, augmented sequence of speaker labels
    """
    # each entry in array is list of speakers in a meeting
    speaker_labels_array = np.array(list(segmented_speakers_dict.values()), dtype=list)
    # choose random sequence of speaker labels
    random_speaker_seq = np.random.choice(speaker_labels_array)
    # randomly truncate sequence to meeting length (effectively sub meeting randomisation)
    if meeting_length < len(random_speaker_seq):
        start_index = np.random.choice(len(random_speaker_seq) - meeting_length + 1)
        end_index = start_index + meeting_length - 1
        random_speaker_seq = random_speaker_seq[start_index:end_index+1]
    else:  # TODO: this needs to be improved. just for testing. shortest train meeting is 71
        print("shorter meeting returned")
    # choose meeting to sample from, ensure it has at least the same number of speakers
    num_speakers = len(set(random_speaker_seq))
    valid_meeting_ids = np.array(list(meeting_dvec_dict.keys()))
    indexes_to_remove = []
    for i, meeting_id in enumerate(valid_meeting_ids):  #TODO: this only needs to be done once
        if len(set(segmented_speakers_dict[meeting_id])) < num_speakers:
            indexes_to_remove.append(i)
    valid_meeting_ids = np.delete(valid_meeting_ids, indexes_to_remove)
    random_meeting_id = np.random.choice(valid_meeting_ids)
    # create set of current unique speakers in sequence
    current_speakers = set(random_speaker_seq)
    # create set of unique speakers available from new meeting
    new_speakers = set(segmented_speakers_dict[random_meeting_id])
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
        random_idx = np.random.choice(len(meeting_dvec_dict[random_meeting_id][speaker]))
        random_dvec = meeting_dvec_dict[random_meeting_id][speaker][random_idx]
        augmented_meeting.append(random_dvec)
    return augmented_meeting, random_speaker_seq


def Diaconis(batch):
    """Randomly samples a rotation matrix and applies to each meeting in a batch of data.
       Returns the rotated batch.

       :param: dict batch[meeting_id] = List[dvectors] (dictionary of meetings, each a list of segs)
       :return: dict batch[meeting_id] = List[dvectors] (as above with but each meeting rotated)
    """
    dimension = 32  # of d-vector
    for meeting_id in batch:
        rotation_mat = SO.rvs(dimension)  # use different rotation matrix for each meeting
        batch[meeting_id] = np.array(batch[meeting_id])
        # rotate meeting
        batch[meeting_id] = np.dot(batch[meeting_id], rotation_mat)
        # normalise variance
        batch[meeting_id] *= np.sqrt(dimension)
    return batch


def produce_augmented_batch_function(dataset='dev', batch_size=25, aug_type="global",
                                        meeting_length=50, Diac=True):  # for debugging only
    """Produces a batch of augmented data for training.
       The dicts contain original meetings.  Only dicts corresponding to aug_types are required.

       :param: str dataset: "train", "dev", or "eval"
       :param: int batch_size: number of augmented meetings to produce
       :param: str aug_type: "global", "meeting" or "None" (randomisation type)
       :param: int meeting_length: number of segments in each augmented meeting
       :param: Bool Diac: performs Diaconis augmentation on top of batch if True
       :return: dict aug_meetings[meeting_id] = List[dvectors]: batch of augmented meetings
       :return: dict aug_speakers[meeting_id] = List[str]: speaker labels sequences
    """
    batch_size = int(batch_size)
    # load data
    averaged_segmented_meetings_dict, segmented_speakers_dict = build_segment_dicts(dataset)
    if aug_type == "global":
        global_dvec_dict = build_global_dvec_dict(dataset)
    elif aug_type == "meeting":
        meeting_dvec_dict = build_meeting_dvec_dict(dataset)

    # Two dictionaries with key as new meeting_id
    aug_meetings = {}  # Value is augmented meeting (1 d-vector per segment)
    aug_speakers = {}  # Value is labels for meeting (1 speaker per segment)

    # if only sub-meeting
    if aug_type == "None":
        # TODO: move for loops to function so don't have to repeat some lines
        for i in range(batch_size):
            aug_meeting_id = "AUG_" + str(i)
            aug_meeting, aug_speaker = sub_meeting_augmentation(averaged_segmented_meetings_dict,
                                                            segmented_speakers_dict, meeting_length)
            aug_meetings[aug_meeting_id] = aug_meeting
            aug_speakers[aug_meeting_id] = aug_speaker
 
    elif aug_type == "global":
        for i in range(batch_size):
            aug_meeting_id = "AUG_" + str(i)
            aug_meeting, aug_speaker = global_speaker_randomisation(global_dvec_dict,
                                                            segmented_speakers_dict, meeting_length)
            aug_meetings[aug_meeting_id] = aug_meeting
            aug_speakers[aug_meeting_id] = aug_speaker

    elif aug_type == "meeting":
        for i in range(batch_size):
            aug_meeting_id = "AUG_" + str(i)
            aug_meeting, aug_speaker = meeting_speaker_randomisation(meeting_dvec_dict,
                                                            segmented_speakers_dict, meeting_length)
            aug_meetings[aug_meeting_id] = aug_meeting
            aug_speakers[aug_meeting_id] = aug_speaker

    else:
        raise ValueError("Invalid aug_type.")

    # do Diac aug on entire batch
    if Diac == True:
        aug_meetings = Diaconis(aug_meetings)
    else:
        for meeting_id in aug_meetings:
            aug_meetings[meeting_id] = np.array(aug_meetings[meeting_id])

    return aug_meetings, aug_speakers


def produce_augmented_batch(dataset='dev', batch_size=25, aug_type="global", meeting_length=50,
        num_batches=int(1e10), Diac=True):  # generator version as used in on-the-fly
    """Generator to produce mini-batches of augmented data for training.
       The dicts contain original meetings.  Only dicts corresponding to aug_types are required.

       :param: str dataset: "train", "dev", or "eval"
       :param: int batch_size: number of augmented meetings to produce
       :param: str aug_type: "global", "meeting" or "None" (randomisation type)
       :param: int meeting_length: number of segments in each augmented meeting
       :param: Bool Diac: performs Diaconis augmentation on top of batch if True
       :yield: List[Tuple[List[List[dvectors]], List[List[str]]]]  batch of augmented meetings and 
        speaker label sequences in format required by ESPNet
    """
    batch_size = int(batch_size)
    # load data
    averaged_segmented_meetings_dict, segmented_speakers_dict = build_segment_dicts(dataset)
    if aug_type == "global":
        global_dvec_dict = build_global_dvec_dict(dataset)
    elif aug_type == "meeting":
        meeting_dvec_dict = build_meeting_dvec_dict(dataset)

    for _ in range(num_batches):
        # Two dictionaries with key as new meeting_id
        aug_meetings = {}  # Value is augmented meeting (1 d-vector per segment)
        aug_speakers = {}  # Value is labels for meeting (1 speaker per segment)

        # if sub-meeting
        if aug_type == "None":
            for i in range(batch_size):
                aug_meeting_id = "AUG_" + str(i)
                # TODO: move for loops to function so don't have to repeat some lines
                aug_meeting, aug_speaker = sub_meeting_augmentation(averaged_segmented_meetings_dict,
                                                                segmented_speakers_dict, meeting_length)
                aug_meetings[aug_meeting_id] = aug_meeting
                aug_speakers[aug_meeting_id] = aug_speaker
            
        elif aug_type == "global":
            for i in range(batch_size):
                aug_meeting_id = "AUG_" + str(i)
                aug_meeting, aug_speaker = global_speaker_randomisation(global_dvec_dict,
                                                       segmented_speakers_dict, meeting_length)
                aug_meetings[aug_meeting_id] = aug_meeting
                aug_speakers[aug_meeting_id] = aug_speaker

        elif aug_type == "meeting":
            for i in range(batch_size):
                aug_meeting_id = "AUG_" + str(i)
                aug_meeting, aug_speaker = meeting_speaker_randomisation(meeting_dvec_dict,
                                                          segmented_speakers_dict, meeting_length)
                aug_meetings[aug_meeting_id] = aug_meeting
                aug_speakers[aug_meeting_id] = aug_speaker

        else:
            raise ValueError("Invalid aug_type")

        # do Diac aug on entire batch
        if Diac == True:
            aug_meetings = Diaconis(aug_meetings)
        else:
            for meeting_id in aug_meetings:
                aug_meetings[meeting_id] = np.array(aug_meetings[meeting_id])

        # convert aug_meetings/aug_speakers to required format for iterator class
        aug_meetings_list = []
        aug_speakers_list = []
        for meeting_id in aug_meetings:
            aug_meetings_list.append(aug_meetings[meeting_id])

            # convert speaker labels to numbers
            labels = aug_speakers[meeting_id]
            spk_dict = {label: i for i, label in enumerate(set(labels))}
            labels = [spk_dict[label] for label in labels]
            aug_speakers_list.append(np.array(labels))
        batch =[(aug_meetings_list, aug_speakers_list)]
        yield batch


def write_to_json(meetings, speakers, dataset, aug_type):  # for debugging only
    """Write batch to JSON file.

    :param: dict meetings[meeting_id] = List[dvectors] (lists of segments for each meeting)
    :param: dict speakers[meeting_id] = List[str]  (speaker label sequences)
    :param: str dataset: "train", "dev" or "eval"
    :param: str aug_type: "None", "meeting" or "global"
    """ 
    json_dict = {}
    json_dict["utts"] = {}
    with open("/data/mifs_scratch/jhrt2/aug_data_%s/%s.scp" % (aug_type, dataset)) as _scp:
        meeting_level_scp = {eachline.split()[0]:eachline.split()[1].rstrip()
                                for eachline in _scp.readlines()}
    for meeting_id, meeting in meetings.items():
        labels = speakers[meeting_id]
        input_dict = {}
        input_dict["feat"] = meeting_level_scp[meeting_id]
        input_dict["name"] = "input1"
        input_dict["shape"] = meeting.shape

        output_dict = {}
        output_dict["name"] = "target1"
        # assign speakers integers
        spk_dict = {label: str(i) for i, label in enumerate(set(labels))}
        labels = [spk_dict[label] for label in labels]
        output_dict["shape"] = [len(labels), 4+1]  # where does 4+1 come from?
        output_dict["tokenid"] = ' '.join(labels)
        json_dict["utts"][meeting_id] = {}
        json_dict["utts"][meeting_id]["input"] = [input_dict]
        json_dict["utts"][meeting_id]["output"] = [output_dict]
    with open("/data/mifs_scratch/jhrt2/aug_data_%s/%s.json" % (aug_type, dataset), 'wb') as json_file:
        json_file.write(json.dumps(json_dict, indent=4, sort_keys=True).encode('utf_8'))


def write_to_ark(meetings, dataset, aug_type):  # for debugging only
    """Write each meeting to a separate ark file.

    :param: dict meetings[meeting_id] = List[dvectors] (lists of segments for each meeting)
    :param: str dataset: "train", "dev" or "eval"
    :param: str aug_type: "None", "meeting" or "global"
    """
    with kaldiio.WriteHelper('ark,scp:/data/mifs_scratch/jhrt2/aug_data_%s/%s.ark,/data/mifs_scratch/jhrt2/aug_data_%s/%s.scp' \
         % (aug_type, dataset, aug_type, dataset)) as writer:
        for meeting_id in meetings:
            writer(meeting_id, meetings[meeting_id])


def main():
    """Main"""

    # example for producing a batch for debugging.  Delete if using as generator.
    dataset = "train"
    aug_type = "None"
    Diac = False

    aug_meetings, aug_speakers = produce_augmented_batch_function(dataset=dataset,
                                                        batch_size=735000,
                                                        aug_type=aug_type,
                                                        Diac=Diac)

    write_to_ark(aug_meetings, dataset, aug_type)
    write_to_json(aug_meetings, aug_speakers, dataset, aug_type)

if __name__ == '__main__':
    main()
    