"""Run spectral clustering on segmented data"""

import os
import sys
import argparse
#import json
import itertools
import numpy as np
import json
from SpectralCluster.spectralcluster import SpectralClusterer

from data_loading import build_segment_desc_dict, build_segment_dicts, \
    build_global_dvec_dict, open_rttm, get_file_paths, build_meeting_dvec_dict
from data_aug import produce_augmented_batch
from prep_eval_files import prepare_split_eval


def do_spectral_clustering(dvec_list, gauss_blur=1.0, p_percentile=0.95,
                           minclusters=2, maxclusters=4, truek=4, custom_dist=None):
    """Does spectral clustering using SpectralCluster, see import"""
    if minclusters < 1 and maxclusters < 1:
        if truek == 1:
            return [0] * dvec_list.shape[0]
        clusterer = SpectralClusterer(min_clusters=truek, max_clusters=truek,
                                      p_percentile=p_percentile,
                                      gaussian_blur_sigma=gauss_blur, custom_dist=custom_dist)
    else:
        clusterer = SpectralClusterer(min_clusters=minclusters, max_clusters=maxclusters,
                                      p_percentile=p_percentile,
                                      gaussian_blur_sigma=gauss_blur, custom_dist=custom_dist)
    return clusterer.predict(dvec_list)


def permutation_invariant_seqmatch(hypothesis, reference_list):
    """For calculating segment level error rate calculation"""
    num_perm = max(4, len(set(hypothesis)))
    permutations = itertools.permutations(np.arange(num_perm))
    correct = []
    for permutation in permutations:
        mapping = {old:new for old, new in zip(np.arange(num_perm), permutation)}
        correct.append(sum([1 for hyp, ref in zip(hypothesis, reference_list)
                            if mapping[hyp] == ref]))
    return max(correct)


def evaluate_spectralclustering(args, averaged_segmented_meetings_dict, segmented_speakers_dict):
    """Loops through all meetings to call spectral clustering function"""
    total_correct = 0
    total_length = 0
    results_dict = {}
    for meeting_id in averaged_segmented_meetings_dict:
        cur_mat = np.array(averaged_segmented_meetings_dict[meeting_id])
        reference = segmented_speakers_dict[meeting_id]
        # assign unique integer to each speaker label
        ref_dict = {label: i for i, label in enumerate(set(reference))}
        reference = [ref_dict[label] for label in reference]
        if len(reference) == 1:
            results_dict[meeting_id] = [0]
            continue
        try:
            hypothesis = do_spectral_clustering(cur_mat,
                                                gauss_blur=args.gauss_blur,
                                                p_percentile=args.p_percentile,
                                                minclusters=int(args.minMaxK[0]),
                                                maxclusters=int(args.minMaxK[1]),
                                                truek=len(set(reference)),
                                                custom_dist=args.custom_dist)
        except:
            print("ERROR:: %s %s" % (str(reference), str(cur_mat)))
            raise
        results_dict[meeting_id] = hypothesis

        _correct = permutation_invariant_seqmatch(hypothesis, reference)
        total_length += len(reference)
        total_correct += _correct
    percentage_correct = total_correct * 100 / total_length
    print("Total Correct: %s, Total Length: %s, Percentage Correct: %s" %
          (str(total_correct), str(total_length), str(percentage_correct)))
    return results_dict, percentage_correct


# def write_to_rttm(args, results_dict, dataset):
#     """Creates a copy of data rttm file, replacing the speaker label column with cluster label.
#     Also rewrites reference file with <NA> instead of indices and removes filtered segments."""
#     _, rttm_path = get_file_paths(args, dataset)
#     segments_desc_dict, _ = build_segment_desc_dict(rttm_path) # filtered
#     with open(dataset + "_results_spectral.rttm", "w") as results_file, \
#          open(dataset + "_reference_spectral.rttm", "w") as reference_file:
#          for meeting_id, meeting in segments_desc_dict.items():
#             for segment in meeting:
#                 reference_file.write("SPEAKER " + meeting_id + ' 1 ' + str(segment[3]) + ' ' + 
#                                      str(segment[5]) + ' <NA> <NA> ' + segment[2] + ' <NA>\n')
#                 hypothesis = results_dict[meeting_id][0]
#                 results_dict[meeting_id] = np.delete(results_dict[meeting_id], 0)
#                 results_file.write("SPEAKER " + meeting_id + ' 1 ' + str(segment[3]) + ' ' + 
#                                      str(segment[5]) + ' <NA> <NA> ' + str(hypothesis) + ' <NA>\n')


def write_to_json(args, results_dict):
    """Write results to JSON file.

    """
    json_dict = {}
    json_dict["utts"] = {}
    for meeting_id, speaker_labels in results_dict.items():
        output_dict = {}
        output_dict["name"] = "target1"
        output_dict["shape"] = [len(speaker_labels), 4+1]  # where does 4+1 come from?
        speaker_labels = list(speaker_labels)
        for i in range(len(speaker_labels)):
            speaker_labels[i] = str(speaker_labels[i])
        speaker_labels.append('4')  # end of sequence token
        output_dict["rec_tokenid"] = ' '.join(speaker_labels)
        json_dict["utts"][meeting_id] = {}
        json_dict["utts"][meeting_id]["output"] = [output_dict]
    with open(args.output_path, 'wb') as json_file:
        json_file.write(json.dumps(json_dict, indent=4, sort_keys=True).encode('utf_8'))


def setup():
    """Get cmds and setup directories."""
    cmdparser = argparse.ArgumentParser(description='Do speaker clsutering based on'\
                                                    'refined version of spectral clustering',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdparser.add_argument('--gauss-blur', help='gaussian blur for spectral clustering',
                           type=float, default=0.1)
    cmdparser.add_argument('--p-percentile', help='p_percentile for spectral clustering',
                           type=float, default=0.93)
    cmdparser.add_argument('--custom-dist', help='e.g. euclidean, cosine', type=str, default=None)
    # cmdparser.add_argument('--json-out', dest='output_json',
    #                        help='json output file used for scoring', default=None)
    cmdparser.add_argument('--minMaxK', nargs=2, default=[2, 4])

    # cmdparser.add_argument('--eval-emb', type=str,
    #         default="/home/mifs/jhrt2/newDNC/data/arks.concat/eval.scp", help='')
    # cmdparser.add_argument('--eval-rttm', type=str,
    #         default="/home/mifs/jhrt2/newDNC/data/rttms.concat/eval.rttm", help='')
    # cmdparser.add_argument('--eval-emb', type=str,
    #         default="/home/mifs/jhrt2/newDNC/data/arks.meeting.cmn.tdnn/eval.scp", help='')
    cmdparser.add_argument('--eval-emb', type=str,
            default="/data/mifs_scratch/jhrt2/james/eval150", help='')
    cmdparser.add_argument('--eval-rttm', type=str,
            default="/home/mifs/jhrt2/newDNC/data/window_level_rttms/eval150_window_level.rttm", help='')
    cmdparser.add_argument('--tdoa-directory', type=str,
            default="/data/mifs_scratch/jhrt2/BeamformIt/MDM_AMI_fixedref_10", help='')
    #cmdparser.add_argument('injson', help='ark files containing the meetings', type=str)
    cmdparser.add_argument('--output-path', type=str,
            default="/data/mifs_scratch/jhrt2/models/FinalResults/spectral/window150wav2vec2gccphat/eval95k24.1.json", help='')
    cmdargs = cmdparser.parse_args()
    return cmdargs


def main():
    """main"""
    # should use reference from gen_reference at window-level
    # makes json so after this just evaluate same way as DNC (run prep_eval_files)
    # remember to edit arg paths
    dataset = "eval"
    args = setup()
    emb = "wav2vec2"
    tdoa = False
    gccphat = True
    meeting_length = 101

    meetings, speakers = build_segment_dicts(args, dataset, emb=emb, tdoa=tdoa, gccphat=gccphat)

    split_meetings, split_speakers = prepare_split_eval(meetings, speakers, meeting_length=meeting_length)
    results_dict, _ = evaluate_spectralclustering(args, split_meetings, split_speakers)

    #write_to_rttm(args, results_dict, dataset)

    write_to_json(args, results_dict)


if __name__ == '__main__':
    main()
    