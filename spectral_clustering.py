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
# import pickle
from data_loading import build_segment_dicts, build_global_dvec_dict, open_rttm

def setup():
    """Get cmds and setup directories."""
    cmdparser = argparse.ArgumentParser(description='Do speaker clsutering based on'\
                                                    'refined version of spectral clustering',
                                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    cmdparser.add_argument('--gauss-blur', help='gaussian blur for spectral clustering',
                           type=float, default=0.1)
    cmdparser.add_argument('--p-percentile', help='p_percentile for spectral clustering',
                           type=float, default=0.95)
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

# def load_obj(name):
#     """Loads an object from /obj using pickle."""
#     with open('obj/' + name + '.pkl', 'rb') as f:
#         return pickle.load(f)

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
        ref_dict = {label: i for i, label in enumerate(set(reference))}
        reference = [ref_dict[label] for label in reference]
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
        _correct = permutation_invariant_seqmatch(hypothesis, reference)
        total_length += len(reference)
        total_correct += _correct
    print("Total Correct: %s, Total Length: %s, Percentage Correct: %s" %
          (str(total_correct), str(total_length), str(total_correct * 100 / total_length)))
    return results_dict


# def write_results_dict(results_dict, output_json):
#     """Writes the results dictionary into json file"""
#     output_dict = {"utts":{}}
#     for meeting_name, hypothesis in results_dict.items():
#         hypothesis = " ".join([str(i) for i in hypothesis]) + " 4"
#         output_dict["utts"][meeting_name] = {"output":[{"rec_tokenid":hypothesis}]}
#     with open(output_json, 'wb') as json_file:
#         json_file.write(json.dumps(output_dict, indent=4, sort_keys=True).encode('utf_8'))
#     return

def write_to_rttm(results_dict):  # Hard coded for eval: change 
    """Creates a copy of data rttm file, replacing the speaker label column with cluster label."""
    segments_desc_list = open_rttm("data/rttms.concat/eval.rttm")
    with open("results.rttm", "w") as results_file:
        for segment_desc in segments_desc_list:
            meeting_id = segment_desc[1]
            # change speaker label column (7)
            segment_desc[7] = results_dict[meeting_id][0]
            results_dict[meeting_id] = np.delete(results_dict[meeting_id], 0)
            for i in range(len(segment_desc)-1):
                results_file.write(str(segment_desc[i]) + ' ')
            results_file.write(str(segment_desc[len(segment_desc)-1]))  # no trailing space
            results_file.write('\n')


def main():
    """main"""
    args = setup()
    # averaged_segmented_meetings_dict = load_obj("averaged_segmented_meetings_dict")
    averaged_segmented_meetings_dict, segmented_speakers_dict = build_segment_dicts("eval")
    # global_dvec_dict = build_global_dvec_dict("eval")
    # averaged_segmented_meetings_dict = {}
    # averaged_segmented_meetings_dict['meeting_id'] = global_dvec_dict["MEE071"]
    # segmented_speakers_dict = {}
    # segmented_speakers_dict['meeting_id'] = ["MEE071" for i in range(len(averaged_segmented_meetings_dict['meeting_id']))]
    # segmented_speakers_dict = load_obj("segmented_speakers_dict")
    results_dict = evaluate_spectralclustering(args, averaged_segmented_meetings_dict, segmented_speakers_dict)
    write_to_rttm(results_dict)
    # for key, value in results_dict.items():
        # print(key, value)
    # if args.output_json is not None:
    #     write_results_dict(results_dict, args.output_json)

if __name__ == '__main__':
    main()
    