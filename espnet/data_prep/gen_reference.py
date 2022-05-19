"""Split full rttm file into (sub-)meeting rttm file"""

import argparse
from collections import defaultdict

from data_loading import build_segment_desc_dict

def produce_eval_reference(evaldnc_path, refoutputeval_path):
    """Takes in refoutputeval.rttm and produces reference.rttm ensuring there are splits at meeting
    boundaries in eval.json/evaldnc.rttm"""

    eval_dict, _ = build_segment_desc_dict(evaldnc_path, filt=False)
    start_times_dict = defaultdict(list)  # key is meeting_id, value is list of start times for sub_meetings
    for sub_meeting_id, sub_meeting in eval_dict.items():
        partial_meeting_id = sub_meeting_id[4:12]  # eg. 0EN2002a
        start_times_dict[partial_meeting_id].append(sub_meeting[0][3])

    ref_dict, _ = build_segment_desc_dict(refoutputeval_path, filt=False)
    new_ref = defaultdict(list)
    
    for meeting_id in ref_dict:
        # sort by start time (file is sorted first by speaker)
        ref_dict[meeting_id].sort(key=lambda segment_desc: segment_desc[3])
        partial_meeting_id = meeting_id[7:]
        start_times = start_times_dict[partial_meeting_id]
        for segment_desc in ref_dict[meeting_id]:
            for submeeting_num, start_time in enumerate(start_times):
                if start_time < segment_desc[4] and start_time > segment_desc[3]:
                    # need to split reference if sub_meeting starts within segment
                    # NB: indices are all <NA> so can keep as before
                    first_split_seg = (segment_desc[0], segment_desc[1], segment_desc[2],
                                        segment_desc[3], start_time, round(start_time - segment_desc[3], 2))
                    second_split_seg = (segment_desc[0], segment_desc[1], segment_desc[2],
                                        start_time, segment_desc[4], round(segment_desc[4]-start_time, 2))
                    first_submeeting_id = 'AMI-' + partial_meeting_id + '-' + f"{submeeting_num-1:03d}"
                    second_submeeting_id = 'AMI-' + partial_meeting_id + '-' + f"{submeeting_num:03d}"
                    new_ref[first_submeeting_id].append(first_split_seg)
                    new_ref[second_submeeting_id].append(second_split_seg)
                    break
            else:
                if len(start_times) > 1:
                    for submeeting_num in range(len(start_times)-1):
                        if segment_desc[3] >= start_times[submeeting_num] and segment_desc[3] < start_times[submeeting_num+1]:
                            break
                    else:
                        submeeting_num += 1
                else:
                    submeeting_num = 0
                submeeting_id = 'AMI-' + partial_meeting_id + '-' + f"{submeeting_num:03d}"
                new_ref[submeeting_id].append(segment_desc)

        # # sort meeting by speaker_label, then start_time
        # new_ref[meeting_id].sort(key=lambda segment_desc : (segment_desc[2], segment_desc[3]))

    # write back to rttm
    with open("reference.rttm", "w") as rttm_file:
        for meeting_id, meeting in new_ref.items():
            for segment in meeting:
                rttm_file.write("SPEAKER " + meeting_id + ' 1 ' + str(segment[3]) + ' ' + 
                    str(segment[5]) + ' ' + str(segment[0]) + ' ' + str(segment[1]) + ' ' 
                    + segment[2] + ' <NA>\n')

def main():
    """main procedure"""
    parser = argparse.ArgumentParser(description="split reference file")
    parser.add_argument('--submeeting-rttm', required=True, type=str,
                        help="path to an rttm file that is already split into submeetings")  # should be of form "/home/mifs/jhrt2/newDNC/espnet/egs/ami/dnc1/exp/mdm_train_pytorch_***/decode_mdm_dev_decode/evaldnc.rttm"
    parser.add_argument('--input-rttm', type=str,
                        default = "/home/mifs/jhrt2/newDNC/data/eval_silence_stripped_reference.rttm",
                        help="path to an input ref rttm file")
    args = parser.parse_args()

    produce_eval_reference(args.submeeting_rttm, args.input_rttm)


if __name__ == '__main__':
    main()
