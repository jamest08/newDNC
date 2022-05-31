"""SEGMENT ALIGNED"""
"""Splits meeting into 1s windows (NOT speaker-homogeneous segnents) and saves to rttm file"""

from math import ceil
from collections import defaultdict

from data_loading import build_segment_desc_dict

dataset = "train"
# use silence stripped rttms for dev, eval (ss_{dataset}.rttm)
#rttm_path = "/home/mifs/jhrt2/newDNC/data/rttms/ss_eval.rttm"
rttm_path = "/home/mifs/jhrt2/newDNC/data/rttms/test_train.rttm"

if dataset == 'eval' or dataset == 'dev':
    filt = False  # important not to filter segments so timings stay correct to avoid missed speech
else:
    filt = True
segment_desc_dict, _ = build_segment_desc_dict(rttm_path, filt=filt)

"""
For each meeting, create a list of windows where the shift between window centres is ~1s.  A sequence of windows
should cover a segment of one or more speakers (ie. a period without silence), aligned to start and
end with the segment boundaries.  To allow this, windows may be slightly less than 1s.
See 'Cosine-Distance Virtual Adversarial Training for Semi-Supervised Speaker-Discriminative
Acoustic Embeddings' Section 3.2 for further details.
Ignore windows with mostly silence.
Speaker label is the label which covers most of the window (if equal, randomise).
Output a new rttm file where each line is one window.  Requires start_index, end_index, speakerlabel
Start and end times correspond to the period for which each point in time is closest to the centre 
of that window, or the start/end of the combined segment.

"new_segments_desc_dict[meeting_id] = List[List[start_index, end_index, speaker_label, start_time,
end_time, duration]]

NB: I know this script is inefficient but it only needs to be run once and makes it clear what's 
happening at each stage and so is easy to edit.
"""

new_segments_desc_dict = defaultdict(list)
frames_per_dvector = 15  # the number of frames used to generate one d-vector
desired_window_length = 150

# first just establishing window start and end indices
for meeting_id, meeting in segment_desc_dict.items():
    # sort by segment start index (train rttm not already like this)
    meeting.sort(key=lambda segment: segment[0])
    # find start and end indices of combined segment
    # segment_index is the line in the file for that segment
    segment_index = 0
    while segment_index < len(meeting):
        windows = []  # List[((start_index, end_index, speaker_label, start_time, end_time, duration))] for combined segment
        combined_segment_start_index = meeting[segment_index][0]
        combined_segment_start_time = meeting[segment_index][3]  # times not exactly aligned with indices
        combined_segment_end_index = meeting[segment_index][1]
        combined_segment_end_time = meeting[segment_index][4]
        while segment_index + 1 < len(meeting):
            segment_index += 1
            next_segment_start_index = meeting[segment_index][0]
            next_segment_end_index = meeting[segment_index][1]
            next_segment_end_time = meeting[segment_index][4]
            if next_segment_start_index <= combined_segment_end_index:
                if next_segment_end_index > combined_segment_end_index:
                    combined_segment_end_index = next_segment_end_index
                    combined_segment_end_time = next_segment_end_time
            else:
                segment_index -= 1
                break

        # find start and end indices
        combined_segment_length = combined_segment_end_index-combined_segment_start_index + 1
        if combined_segment_length <= frames_per_dvector:
            # in this case, just have one dvector for the segment, at middle of seg
            window_start_index = combined_segment_start_index + combined_segment_length//2
            window_end_index = window_start_index + 1  # end_index is non inclusive
            windows.append([window_start_index, window_end_index, "", 0, 0, 0])
        else:
            # have as many dvectors as will fit
            num_windows = ceil((combined_segment_length - frames_per_dvector)/desired_window_length)
            window_length = (combined_segment_length - frames_per_dvector)/ num_windows
            window_start_index = combined_segment_start_index + frames_per_dvector // 2
            window_end_index = window_start_index + window_length
            rounded_window_start_index = int(round(window_start_index))
            rounded_window_end_index = int(round(window_end_index))
            windows.append([rounded_window_start_index, rounded_window_end_index, " ", 0, 0, 0])
            for i in range(num_windows-1):
                window_start_index = window_end_index
                window_end_index =  window_start_index + window_length
                rounded_window_start_index = int(round(window_start_index)) + 1
                rounded_window_end_index = int(round(window_end_index))
                windows.append([rounded_window_start_index, rounded_window_end_index, "", 0, 0, 0])

        # now define start and end times based on centre of indices
        # first window start time
        window_index = 0
        windows[window_index][3] = combined_segment_start_time
        for window_index in range(1, len(windows)):
            prev_centre = (windows[window_index-1][0] +
                windows[window_index-1][1]) // 2
            current_centre = (windows[window_index][0] +
                windows[window_index][1]) // 2
            # the centre of the two centres is the time boundary
            boundary = (prev_centre + current_centre) // 2
            # prev end time
            windows[window_index-1][4] = round(boundary/100, 2)
            # prev duration
            windows[window_index-1][5] = round(windows[window_index-1][4] - windows[window_index-1][3], 2)
            # current start time = prev end
            windows[window_index][3] = round(boundary/100, 2)
        # final window end time
        windows[window_index][4] = combined_segment_end_time
        # final window duration
        windows[window_index][5] = round(windows[window_index][4] - windows[window_index][3], 2)

        new_segments_desc_dict[meeting_id] += windows
        segment_index += 1


# now find speaker label for each window
for meeting_id, meeting in segment_desc_dict.items():
    segment_index = 0
    window_index = 0
    final_meeting_index = meeting[-1][1]
    while window_index < len(new_segments_desc_dict[meeting_id]):
        window_start_index = new_segments_desc_dict[meeting_id][window_index][0]
        window_end_index = new_segments_desc_dict[meeting_id][window_index][1]
        segment_start_index = meeting[segment_index][0]
        segment_end_index = meeting[segment_index][1]
        # find amount window and segment intersect
        overlap1 = len(set(range(window_start_index, window_end_index+1)).intersection(
            range(segment_start_index, segment_end_index+1)))
        if overlap1 == 0:
            segment_index += 1
        else:
            # check next segment in case segments overlap
            if segment_index + 1 <= len(meeting) - 1:
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
            new_segments_desc_dict[meeting_id][window_index][2] = window_speaker_label
            window_index += 1


#write to rttm
with open(dataset + str(desired_window_length) + "_window_level.rttm", "w") as rttm_file:
    for meeting_id, meeting in new_segments_desc_dict.items():
        for segment in meeting:
            rttm_file.write("SPEAKER " + meeting_id + ' 1 ' + str(segment[3]) + ' ' + 
                str(segment[5]) + ' ' + str(segment[0]) + ' ' + str(segment[1]) + ' ' 
                + segment[2] + ' <NA>\n')
