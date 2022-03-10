"""Produce silence stripped reference rttm"""

from collections import defaultdict
from data_loading import build_segment_desc_dict

ref_path = "/home/mifs/jhrt2/newDNC/data/silence_stripped_reference.rttm"  
new_path = "/home/mifs/jhrt2/newDNC/data/rttms/ss_dev.rttm" # to be used as input rttm (gets index data) in gen window rttm

ref_segments_dict = build_segment_desc_dict(ref_path, filt=False)
new_segments_dict = defaultdict(list)

dvector_shift = 100

for meeting_id, meeting in ref_segments_dict.items():
    for segment in meeting:
        start_index = round(segment[3]*dvector_shift)
        end_index = round(segment[4]*dvector_shift)
        new_segments_dict[meeting_id].append((start_index, end_index, segment[2], segment[3],
                                    segment[4], segment[5]))
    # sort by segment start index (ref rttm sorted by speaker)
    new_segments_dict[meeting_id].sort(key=lambda segment: segment[0])


# write to rttm
with open(new_path, "w") as rttm_file:
    for meeting_id, meeting in new_segments_dict.items():
        for segment in meeting:
            rttm_file.write("SPEAKER " + meeting_id + ' 1 ' + str(segment[3]) + ' ' + 
                str(segment[5]) + ' ' + str(segment[0]) + ' ' + str(segment[1]) + ' ' 
                + segment[2] + ' <NA>\n')
