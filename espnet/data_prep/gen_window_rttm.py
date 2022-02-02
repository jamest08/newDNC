"""Splits meeting into 1s windows (NOT speaker-homogeneous segnents and saves to rttm file"""

import kaldiio
import numpy as np
from collections import defaultdict

from data_loading import build_segment_desc_dict

dataset = "eval"
rttm_path = "/home/dawna/flk24/files4jhrt2/DNC/data/rttms/test_%s.rttm" % dataset  # to be provided by Florian
segment_desc_dict = build_segment_desc_dict(rttm_path)

"""
For each meeting, create a list of windows where each window is 1s long.  Ignore windows with mostly
silence.  Speaker label is the label which covers most of the window (if equal, randomise).
Output a new rttm file where each line is one window.  Requires start_index, end_index, speakerlabel
and convert to time.

If meeting has 5 speakers, create 5 new meetings where 1 speaker is removed in each.
"""

# new_segments_desc_dict[meeting_id] = List((start_index, end_index, speaker_label, start_time,
#   end_time, duration))

new_segments_desc_dict = defaultdict(list)


window_length = 100  # 1 second
for meeting_id, meeting in segment_desc_dict.items():
    segment_index = 0
    window_start_index = 0
    final_meeting_index = meeting[-1][1]
    while True:
        window_end_index = window_start_index + window_length-1
        if window_end_index > final_meeting_index or segment_index >= len(meeting) - 1:
            break
        segment_start_index = meeting[segment_index][0]
        segment_end_index = meeting[segment_index][1]
        # find amount window and segment intersect
        overlap1 = len(set(range(window_start_index, window_end_index+1)).intersection(
            range(segment_start_index, segment_end_index+1)))
        if overlap1 == 0:
            if segment_start_index > window_end_index:
                window_start_index += window_length
            else:
                segment_index += 1
        else:
            if segment_index + 1 <= len(meeting) - 1:
                # check next segment in case segments overlap
                next_segment_start_index = meeting[segment_index+1][0]
                next_segment_end_index = meeting[segment_index+1][1]
                overlap2 = len(set(range(window_start_index, window_end_index+1)).intersection(
                    range(next_segment_start_index, next_segment_end_index+1)))
                if overlap1 > overlap2:  # chose speaker label from first segment
                    window_speaker_label = meeting[segment_index][2]
                else:  # chose speaker label from first segment
                    window_speaker_label = meeting[segment_index+1][2]
            else:
                window_speaker_label = meeting[segment_index][2]
            window_start_time = round(window_start_index/100, 2)
            window_end_time = round(window_end_index/100, 2)
            window_duration = round(window_length/100, 2)
            new_segments_desc_dict[meeting_id].append((window_start_index, window_end_index,
                        window_speaker_label, window_start_time, window_end_time, window_duration))
            window_start_index += window_length

# split meetings with five speakers
for meeting_id, meeting in new_segments_desc_dict.items():
    speaker_list = list(set([segment[2] for segment in meeting]))
    if len(speaker_list) == 5:
        for segment in meeting:
            if segment[2] == speaker_list[0]:
                new_segments_desc_dict[meeting_id + 'a'].append(segment)
                new_segments_desc_dict[meeting_id + 'b'].append(segment)
                new_segments_desc_dict[meeting_id + 'c'].append(segment)
                new_segments_desc_dict[meeting_id + 'd'].append(segment)
            elif segment[2] == speaker_list[1]:
                new_segments_desc_dict[meeting_id + 'a'].append(segment)
                new_segments_desc_dict[meeting_id + 'b'].append(segment)
                new_segments_desc_dict[meeting_id + 'c'].append(segment)
                new_segments_desc_dict[meeting_id + 'e'].append(segment)
            elif segment[2] == speaker_list[2]:
                new_segments_desc_dict[meeting_id + 'a'].append(segment)
                new_segments_desc_dict[meeting_id + 'b'].append(segment)
                new_segments_desc_dict[meeting_id + 'd'].append(segment)
                new_segments_desc_dict[meeting_id + 'e'].append(segment)
            elif segment[2] == speaker_list[3]:
                new_segments_desc_dict[meeting_id + 'a'].append(segment)
                new_segments_desc_dict[meeting_id + 'c'].append(segment)
                new_segments_desc_dict[meeting_id + 'd'].append(segment)
                new_segments_desc_dict[meeting_id + 'e'].append(segment)
            elif segment[2] == speaker_list[4]:
                new_segments_desc_dict[meeting_id + 'b'].append(segment)
                new_segments_desc_dict[meeting_id + 'c'].append(segment)
                new_segments_desc_dict[meeting_id + 'd'].append(segment)
                new_segments_desc_dict[meeting_id + 'e'].append(segment)
        del new_segments_desc_dict[meeting_id]


# write to rttm
with open(dataset + "_window_level.rttm", "w") as rttm_file:
    for meeting_id, meeting in new_segments_desc_dict.items():
        for segment in meeting:
            rttm_file.write("SPEAKER " + meeting_id + ' 1 ' + str(segment[3]) + ' ' + 
                str(segment[5]) + ' ' + str(segment[0]) + ' ' + str(segment[1]) + ' ' 
                + segment[2] + ' <NA>\n')
