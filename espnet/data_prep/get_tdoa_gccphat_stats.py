import numpy as np
import configargparse

from data_loading import build_segment_desc_dict, get_file_paths


def get_tdoagccphat_stats(args, meeting_ids):
    """Returns mean and standard deviation vectors for TDOA and GCC-PHAT in dataset                                         
    """
    directory_path = args.tdoa_directory

    # could calculate mean/std iteratively to save space
    all_tdoa_vecs = []
    all_gccphat_vecs = []

    for meeting_id in meeting_ids:
        partial_meeting_id = meeting_id[8:]
        with open(directory_path + '/' + partial_meeting_id + '.del', 'r') as delays:
            delays_list = [(line.strip()).split() for line in delays]
        for delay in delays_list:
            # ignore second channel as that is fixed reference (always 0 1.000000)
            tdoa_vec = np.array([int(delay[2]), int(delay[6]), int(delay[8]), int(delay[10]),
                                int(delay[12]), int(delay[14]), int(delay[16])])
            gccphat_vec = np.array([float(delay[3]), float(delay[7]), float(delay[9]), float(delay[11]),
                                float(delay[13]), float(delay[15]), float(delay[17])])
            all_tdoa_vecs.append(tdoa_vec)
            all_gccphat_vecs.append(gccphat_vec)

    all_tdoa_vecs = np.array(all_tdoa_vecs, dtype=np.float32)
    all_gccphat_vecs = np.array(all_gccphat_vecs, dtype=np.float32)
    tdoa_mean = np.mean(all_tdoa_vecs, axis=0)  # mean vector 
    gccphat_mean = np.mean(all_gccphat_vecs, axis=0)
    tdoa_std = np.std(all_tdoa_vecs, axis=0)  # standard deviation vector
    gccphat_std = np.std(all_gccphat_vecs, axis=0)

    return tdoa_mean, gccphat_mean, tdoa_std, gccphat_std


def get_parser():  # debugging only, official paths should be maintained in asr_train.py
    parser = configargparse.ArgumentParser(
        description="Load speech data",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train-scp', type=str,
            default="/home/mifs/jhrt2/newDNC/data/arks.meeting.cmn.tdnn/train.scp", help='')
    parser.add_argument('--train-rttm', type=str,
            default="/home/mifs/jhrt2/newDNC/espnet/data_prep/train_window_level.rttm", help='')
    parser.add_argument('--tdoa-directory', type=str,
            default="/data/mifs_scratch/jhrt2/BeamformIt/MDM_AMI_fixedref_10", help='')
    return parser


def main():
    parser = get_parser()
    args, _ = parser.parse_known_args()
    dataset = 'train'
    scp_path, rttm_path = get_file_paths(args, dataset)
    segment_desc_dict = build_segment_desc_dict(rttm_path, filt=True)
    tdoa_mean, gccphat_mean, tdoa_std, gccphat_std = get_tdoagccphat_stats(args, segment_desc_dict.keys())
    print(tdoa_mean, gccphat_mean, tdoa_std, gccphat_std)

if __name__ == '__main__':
    main()