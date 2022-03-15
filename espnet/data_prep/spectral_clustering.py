"""Run spectral clustering on segmented data"""

import os
import sys
import argparse
#import json
import itertools
import numpy as np
from SpectralCluster.spectralcluster import SpectralClusterer

from data_loading import build_segment_desc_dict, build_segment_dicts, \
    build_global_dvec_dict, open_rttm, get_file_paths, build_meeting_dvec_dict
from data_aug import produce_augmented_batch

def setup():
    """Get cmds and setup directories."""
    cmdparser = argparse.ArgumentParser(description='Do speaker clsutering based on'\
                                                    'refined version of spectral clustering',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdparser.add_argument('--gauss-blur', help='gaussian blur for spectral clustering',
                           type=float, default=0.1)
    cmdparser.add_argument('--p-percentile', help='p_percentile for spectral clustering',
                           type=float, default=0.94)
    cmdparser.add_argument('--custom-dist', help='e.g. euclidean, cosine', type=str, default=None)
    # cmdparser.add_argument('--json-out', dest='output_json',
    #                        help='json output file used for scoring', default=None)
    cmdparser.add_argument('--minMaxK', nargs=2, default=[2, 4])
    cmdparser.add_argument('--train-scp', type=str,
            default="/home/mifs/jhrt2/newDNC/data/arks.meeting.cmn/train.scp", help='')
    cmdparser.add_argument('--valid-scp', type=str,
            default="//home/mifs/jhrt2/newDNC/data/arks.meeting.cmn/dev.scp", help='')
    cmdparser.add_argument('--train-rttm', type=str,
            default="/home/mifs/jhrt2/newDNC/data/rttms/test_train.rttm", help='')
    cmdparser.add_argument('--valid-rttm', type=str,
            default="/home/mifs/jhrt2/newDNC/data/rttms/test_dev.rttm", help='')

    cmdparser.add_argument('--eval-scp', type=str,
            default="/home/mifs/jhrt2/newDNC/data/arks.meeting.cmn.tdnn/eval.scp", help='')
    cmdparser.add_argument('--eval-rttm', type=str,
            default="/home/mifs/jhrt2/newDNC/data/rttms/test_eval.rttm", help='')

    #cmdparser.add_argument('injson', help='ark files containing the meetings', type=str)
    cmdargs = cmdparser.parse_args()
    return cmdargs


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


def write_to_rttm(args, results_dict, dataset):
    """Creates a copy of data rttm file, replacing the speaker label column with cluster label.
    Also rewrites reference file with <NA> instead of indices and removes filtered segments."""
    _, rttm_path = get_file_paths(args, dataset)
    segments_desc_list = open_rttm(rttm_path)  # non-filtered
    segments_desc_dict = build_segment_desc_dict(rttm_path) # filtered
    with open(dataset + "_results_spectral.rttm", "w") as results_file, \
         open(dataset + "_reference_spectral.rttm", "w") as reference_file:
         for meeting_id, meeting in segments_desc_dict.items():
            for segment in meeting:
                reference_file.write("SPEAKER " + meeting_id + ' 1 ' + str(segment[3]) + ' ' + 
                                     str(segment[5]) + ' <NA> <NA> ' + segment[2] + ' <NA>\n')
                hypothesis = results_dict[meeting_id][0]
                results_dict[meeting_id] = np.delete(results_dict[meeting_id], 0)
                results_file.write("SPEAKER " + meeting_id + ' 1 ' + str(segment[3]) + ' ' + 
                                     str(segment[5]) + ' <NA> <NA> ' + str(hypothesis) + ' <NA>\n')


def main():
    """main"""
    dataset = "eval"
    args = setup()
    dvec = False
    tdoa = True
    gccphat = False

    averaged_segmented_meetings_dict, segmented_speakers_dict = build_segment_dicts(args, dataset, dvec=dvec, tdoa=tdoa, gccphat=gccphat)

    results_dict, _ = evaluate_spectralclustering(args, averaged_segmented_meetings_dict, segmented_speakers_dict)

    write_to_rttm(args, results_dict, dataset)


if __name__ == '__main__':
    main()
    