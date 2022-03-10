"""Debugging code to find where missed speech is coming from."""

from data_loading import build_segment_desc_dict
import numpy as np

evaldnc_path = '/home/mifs/jhrt2/newDNC/espnet/egs/ami/dnc1/exp/mdm_train_pytorch_windowtdnn/decode_mdm_dev_decode/evaldnc.rttm'
ref_path = '/home/mifs/jhrt2/newDNC/espnet/data_prep/reference.rttm'

evaldnc_dict = build_segment_desc_dict(evaldnc_path, filt='False')
ref_dict = build_segment_desc_dict(ref_path, filt='False')

for meeting_id in evaldnc_dict:
    evaldnc_list = evaldnc_dict[meeting_id]
    ref_list = ref_dict[meeting_id]

    ref_active_time = set()  # set of active times in reference to 2d.p.

    for segment in ref_list:
        start_time = segment[3]
        end_time = segment[4]
        times = set()
        for time in np.arange(start_time, end_time+0.01, 0.01):
            times.add(round(time, 2))
        ref_active_time |= times

    eval_active_time = set()  # set of active times in evaldnc to 2d.p.

    for segment in evaldnc_list:
        start_time = segment[3]
        end_time = segment[4]
        times = set()
        for time in np.arange(start_time, end_time+0.01, 0.01):
            times.add(round(time, 2))
        eval_active_time |= times


    missed_speech_times = ref_active_time - eval_active_time
    print(sorted(list(missed_speech_times)))


