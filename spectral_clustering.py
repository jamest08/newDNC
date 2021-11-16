"""Run spectral clustering on segmented data"""

import os
import sys
import argparse
#import json
import itertools
import numpy as np
from tqdm import tqdm
import kaldiio
import utils
from SpectralCluster.spectralcluster import SpectralClusterer

from data_loading import build_segment_desc_dict, build_segment_dicts, build_global_dvec_dict, \
                         open_rttm, get_file_paths

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
    #cmdparser.add_argument('injson', help='ark files containing the meetings', type=str)
    cmdargs = cmdparser.parse_args()
    # setup output directory and cache commands
    # if cmdargs.output_json is not None:
    #     outdir = os.path.dirname(cmdargs.output_json)
    #     utils.check_output_dir(outdir, True)
    #     utils.cache_command(sys.argv, outdir)
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
    # with open(args.injson) as _json_file:
    #     json_file = json.load(_json_file)
    results_dict = {}
    # for midx, meeting in tqdm(list(json_file["utts"].items())):
    for meeting_id in averaged_segmented_meetings_dict:
        # meeting_input = meeting["input"]
        # meeting_output = meeting["output"]
        # assert len(meeting_input) == 1
        # assert len(meeting_output) == 1
        # meeting_input = meeting_input[0]
        # meeting_output = meeting_output[0]
        # cur_mat = kaldiio.load_mat(meeting_input["feat"])#(samples,features)
        cur_mat = np.array(averaged_segmented_meetings_dict[meeting_id])
        # reference = meeting_output["tokenid"].split()
        # reference = [int(ref) for ref in reference]
        reference = segmented_speakers_dict[meeting_id]
        # assign unique integer to each speaker label
        #print("meeting_id: ", meeting_id, '\n')
        #print("labels: ", reference)
        ref_dict = {label: i for i, label in enumerate(set(reference))}
        reference = [ref_dict[label] for label in reference]
        #print("numbers: ", reference)
        #assert len(reference) == len(cur_mat)
        if len(reference) == 1:
            # results_dict[midx] = [0]
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
        # results_dict[midx] = hypothesis
        results_dict[meeting_id] = hypothesis
        #print("hypothesis", hypothesis)
        _correct = permutation_invariant_seqmatch(hypothesis, reference)
        total_length += len(reference)
        total_correct += _correct
    percentage_correct = total_correct * 100 / total_length
    print("Total Correct: %s, Total Length: %s, Percentage Correct: %s" %
          (str(total_correct), str(total_length), str(percentage_correct)))
    return results_dict, percentage_correct


def write_to_rttm(results_dict, dataset):
    """Creates a copy of data rttm file, replacing the speaker label column with cluster label.
    Also rewrites reference file with <NA> instead of indices and removes filtered segments."""
    _, rttm_path = get_file_paths(dataset)
    segments_desc_list = open_rttm(rttm_path)  # non-filtered
    segments_desc_dict = build_segment_desc_dict(rttm_path) # filtered
    with open(dataset + "_results.rttm", "w") as results_file, \
         open(dataset + "_reference.rttm", "w") as reference_file:
         for meeting_id, meeting in segments_desc_dict.items():
            for segment in meeting:
                reference_file.write("SPEAKER " + meeting_id + ' 1 ' + str(segment[3]) + ' ' + 
                                     str(segment[5]) + ' <NA> <NA> ' + segment[2] + ' <NA>\n')
                hypothesis = results_dict[meeting_id][0]
                results_dict[meeting_id] = np.delete(results_dict[meeting_id], 0)
                results_file.write("SPEAKER " + meeting_id + ' 1 ' + str(segment[3]) + ' ' + 
                                     str(segment[5]) + ' <NA> <NA> ' + str(hypothesis) + ' <NA>\n')
        # for segment_desc in segments_desc_list:
        #     meeting_id = segment_desc[1]
        #     segment_desc[5] = "<NA>"
        #     segment_desc[6] = "<NA>"
        #     for i in range(7):
        #         results_file.write(str(segment_desc[i]) + ' ')
        #         reference_file.write(str(segment_desc[i]) + ' ')
        #     reference_file.write(str(segment_desc[7]) + ' ')
        #     # change speaker label column (7) for results_dict
        #     segment_desc[7] = results_dict[meeting_id][0]
        #     results_dict[meeting_id] = np.delete(results_dict[meeting_id], 0)
        #     results_file.write(str(segment_desc[7]) + ' ')
        #     results_file.write("<NA>")
        #     reference_file.write("<NA>")
        #     results_file.write('\n')
        #     reference_file.write('\n')



def main():
    """main"""
    # optimising parameters
    # dataset = "dev"
    # averaged_segmented_meetings_dict, segmented_speakers_dict = build_segment_dicts(dataset)
    # args = setup()
    # for p_percentile in [0.8, 0.85, 0.87, 0.88, 0.89]:
    #     print(p_percentile)
    #     args.p_percentile = p_percentile
    #     results_dict, percentage_correct = evaluate_spectralclustering(args, averaged_segmented_meetings_dict, segmented_speakers_dict)

    dataset = "eval"
    args = setup()
    averaged_segmented_meetings_dict, segmented_speakers_dict = build_segment_dicts(dataset)
    results_dict, _ = evaluate_spectralclustering(args, averaged_segmented_meetings_dict, segmented_speakers_dict)
    write_to_rttm(results_dict, dataset)


if __name__ == '__main__':
    main()
    