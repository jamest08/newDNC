"""Produces sets of unique speaker label sequences from a batch of aug data."""

import json
import numpy as np

from data_loading import build_segment_desc_dict

def find_set_difference():
    # first find expected size of speaker label set (number of possible submeetings of length 50)
    segment_desc_dict = build_segment_desc_dict('/home/mifs/jhrt2/newDNC/data/rttms.concat/train.rttm', filt=True)
    sub_meeting_count = 0
    for segment_desc_list in segment_desc_dict.values():
        sub_meeting_count += (len(segment_desc_list) - 49)

        
    my_json_path = '/data/mifs_scratch/jhrt2/aug_data_None/train.json'
    orig_json_path = '/home/mifs/jhrt2/DNC/DNC/m50.real.augment/train.json'

    my_label_set = set()
    orig_label_set = set()

    with open(my_json_path, 'r') as my_json_file:
        my_json = json.load(my_json_file)

    with open(orig_json_path, 'r') as orig_json_file:
        orig_json = json.load(orig_json_file)

    print("num my aug meetings: ", len(list(my_json["utts"].values())))
    for my_meeting in my_json["utts"].values():
        my_speaker_sequence = my_meeting["output"][0]["tokenid"]
        my_label_set.add(my_speaker_sequence)

    print("num orig aug meetings: ", len(list(orig_json["utts"].values())))
    for orig_meeting in orig_json["utts"].values():
        orig_speaker_sequence = orig_meeting["output"][0]["tokenid"]
        orig_label_set.add(orig_speaker_sequence)

    print('expected set length: ', sub_meeting_count)
    print("my set length: ", len(my_label_set))
    print("orig set length: ", len(orig_label_set))

    set_difference =  my_label_set - orig_label_set

    print("length of set difference: ", len(set_difference))
    return set_difference

def main():
    """Main"""
    find_set_difference()


if __name__ == '__main__':
    main()