"""Converts eval files to correct format for decoding and scoring."""

import kaldiio
import configargparse
import numpy as np

from data_loading import build_segment_desc_dict, build_segment_dicts, get_file_paths
from data_aug import write_to_ark, write_to_json


def prepare_split_eval(meetings, speakers, meeting_length=100):
    """Split eval set into meetings of equal length, keeping order.  Correct format for decoding.
    Write to json after calling this function.
    
    :param: dict meetings[meeting_id] = List[dvectors] (lists of segments for each meeting)
    :param: dict speakers[meeting_id] = List[str]  (speaker label sequences)
    :param: int number of segments to split meetings into

    :return: dict meetings[split_meeting_id] = List[dvectors]: split meetings
    :return: dict speakers[split_meeting_id] = List[str]: split speaker sequences
    """

    split_meetings = {}
    split_speakers = {}
    for meeting_id, meeting in meetings.items():
        split_meeting_num = 0
        segment_index = 0
        while segment_index + meeting_length <= len(meeting):
            # new meeting id removes MDM and appends split meeting digits with preceding zeros
            split_meeting_id = meeting_id[:3] + meeting_id[6:] + '-' + f"{split_meeting_num:03d}"
            split_meetings[split_meeting_id] = meeting[segment_index:segment_index+meeting_length]
            split_speakers[split_meeting_id] = speakers[meeting_id][segment_index:segment_index+meeting_length]
            segment_index += meeting_length
            split_meeting_num += 1
        if segment_index < len(meeting):
            # final split meeting may be shorter than meeting_length
            split_meeting_id = meeting_id[:3] + meeting_id[6:] + '-' + f"{split_meeting_num:03d}"
            split_meetings[split_meeting_id] = meeting[segment_index:]
            split_speakers[split_meeting_id] = speakers[meeting_id][segment_index:]

    return split_meetings, split_speakers


def produce_eval_scp(meetings, speakers, segment_desc_dict, dataset='eval'):
    """Produces eval.scp in format required for scoring script (each line is one window).
    
    :param: dict meetings[meeting_id] = List[dvectors] (lists of segments for each meeting)
    :param: dict segment_desc_dict[meeting_id] =  List(Tuple(start_index, end_index, speaker_label,
    start_time, end_time, duration))
    """
    scp_meetings = {}
    scp_speakers = {}
    if dataset == 'eval':
        # this matches to existing scp
        speaker_mapping = {'MEE073': 174, 'FEO070': 175, 'FEO072': 176, 'MEE071': 177, 'MEO015': 178, \
            'FEE013': 179, 'MEE014': 180, 'FEE016': 181, 'FIE088': 182, 'FIO087': 183, 'FIO084': 184, \
            'FIO089': 185, 'MTD009PM': 186, 'MTD011UID': 187, 'MTD0010ID': 188, 'MTD012ME': 189}
    elif dataset == 'dev':
        speaker_mapping = {}
        current_speaker_num = 174
    for meeting_id, meeting in meetings.items():
        for segment_index in range(len(meeting)):
            start_time = round(segment_desc_dict[meeting_id][segment_index][3] * 100)
            end_time = round(segment_desc_dict[meeting_id][segment_index][4] * 100)
            speaker = speakers[meeting_id][segment_index]
            try:
                speaker_number = speaker_mapping[speaker]
            except:
                speaker_mapping[speaker] = current_speaker_num
                current_speaker_num += 1
                speaker_number = speaker_mapping[speaker]
            scp_meeting_id = 'AMIXXX-' + f"{speaker_number:05d}" + '-' + meeting_id[7:] + '-XXXXXX-11_XXXXXXX_' + \
                f"{start_time:07d}" + "_" + f"{end_time:07d}"
            scp_speakers[scp_meeting_id] = speaker_number
            scp_meetings[scp_meeting_id] = meeting[segment_index]
    with kaldiio.WriteHelper('ark,scp:/home/mifs/jhrt2/newDNC/scoring/scoring_%s.ark,/home/mifs/jhrt2/newDNC/scoring/scoring_%s.scp' % (dataset, dataset)) as writer:
        for speaker_number in sorted(speaker_mapping.values()):
            for scp_meeting_id in scp_meetings:
                if scp_speakers[scp_meeting_id] == speaker_number:
                    writer(scp_meeting_id, scp_meetings[scp_meeting_id])


def get_parser():  # official paths should be maintained in asr_train.py
    parser = configargparse.ArgumentParser(
        description="Prepare eval files",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--eval-emb', type=str,
            default="/home/mifs/jhrt2/newDNC/data/arks.meeting.cmn.tdnn/eval.scp", help='')
    # parser.add_argument('--eval-emb', type=str,
    #         default="/home/mifs/jhrt2/newDNC/data/arks.concat/eval.scp", help='')
    parser.add_argument('--eval-rttm', type=str,
            default="/home/mifs/jhrt2/newDNC/data/window_level_rttms/eval150_window_level.rttm", help='')
    # parser.add_argument('--eval-emb', type=str,
    #         default="/data/mifs_scratch/jhrt2/james/eval150", help='')
    # parser.add_argument('--eval-rttm', type=str,
    #         default="/home/mifs/jhrt2/newDNC/data/rttms.concat/eval.rttm", help='')

    parser.add_argument('--valid-emb', type=str,
            default="/home/mifs/jhrt2/newDNC/data/arks.meeting.cmn.tdnn/dev.scp", help='')
    # note there are two dev window level rttms.  Here using the silence stripped version
    parser.add_argument('--valid-rttm', type=str,
            default="/home/mifs/jhrt2/newDNC/data/rttms.concat/dev.rttm", help='')
    parser.add_argument('--tdoa-directory', type=str,
            default="/data/mifs_scratch/jhrt2/BeamformIt/MDM_AMI_fixedref_10", help='')
    return parser


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    dataset = 'eval'  # NB: IF DO DEV, REMEMBER NOT DOING VAR NORMALISATION IN DATA_LOADING
    scp_path, rttm_path = get_file_paths(args, dataset)

    meetings, speakers = build_segment_dicts(args, dataset, emb="dvec", tdoa=True, gccphat=True, tdoa_norm=False)
    for meeting_id in meetings:
        meetings[meeting_id] = np.array(meetings[meeting_id])

    meeting_length = 3385

    segment_desc_dict, _ = build_segment_desc_dict(rttm_path)

    produce_eval_scp(meetings, speakers, segment_desc_dict, dataset)

    split_meetings, split_speakers = prepare_split_eval(meetings, speakers, meeting_length)
    write_to_ark(split_meetings, dataset, "None")
    write_to_json(split_meetings, split_speakers, dataset, "None")


if __name__ == '__main__':
    main()
