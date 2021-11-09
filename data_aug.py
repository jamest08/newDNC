"""Augment data on-the-fly"""

import kaldiio
import numpy as np
import os
import sys
import argparse
import json
import multiprocessing as mp
from collections import OrderedDict

import numpy as np
from tqdm import tqdm
import kaldiio
import utils

IDPOS = 2
MAXLOOPITERATIONS = 150
EPS = 10e-15
np.random.seed(0)

def setup():
    """Get cmds and setup directories."""
    parser = argparse.ArgumentParser(description='Prepare Data for Neural Speaker Clustering',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-scps', dest='inscps', action='append', type=str,
                        help='scp files of input data "train.scp eval.scp dev.scp xxx.mlf"')
    parser.add_argument('--input-mlfs', dest='inmlfs', action='append', type=str,
                        help='MLF files of the input data "train.mlf eval.mlf dev.mlf xxx.mlf"')
    parser.add_argument('--filtEncomp', default=False, action='store_true',
                        help='Delete segments encompassed by another')
    parser.add_argument('--maxlen', type=int, default=None,
                        help='maximum input length')
    parser.add_argument('--variableL', nargs=2, type=float, default=None,
                        help='min and max percentages of sequence length'\
                             'between which uniform sampling is used')
    parser.add_argument('--augment', type=int, default=0,
                        help='how many times to augment (0 means no augment, used for evaluation)')
    parser.add_argument('--evensplit', default=False, action='store_true',
                        help='split of meetings will be into equal chunks'\
                        'cannot be used together with variableL'\
                        'augment has to be 0, maxlen has to be set')
    parser.add_argument('--dvectordict', type=str, default=None, action=utils.Abspath,
                        help='dictionary of set of dvectors to be used')
    parser.add_argument('--randomspeaker', default=False, action='store_true',
                        help='for each meeting randomise which speakers to use'\
                        'requires dvectordict')
    parser.add_argument('--maxprocesses', type=int, default=1,
                        help='number of processes in parallel')
    parser.add_argument('--varnormalise', default=False, action='store_true',
                        help='Does variance normalisation by multiplying by sqrt(feature_dim)')
    parser.add_argument('outdir', type=str, action='store',
                        help='Output Directory for the Data')
    cmdargs = parser.parse_args()

    # ensure list of scps and mlfs has the same length
    if len(cmdargs.inscps) != len(cmdargs.inmlfs):
        utils.print_error("number of input scps files and input mlfs has to be the same")
    for idx, scp in enumerate(cmdargs.inscps):
        cmdargs.inscps[idx] = utils.get_abs_path(scp)
        if not scp.endswith('.scp'):
            utils.print_error('scp path has to end with .scp')
    for idx, mlf in enumerate(cmdargs.inmlfs):
        cmdargs.inmlfs[idx] = utils.get_abs_path(mlf)
        if not mlf.endswith('.mlf'):
            utils.print_error('mlf path has to end with .mlf')
    if cmdargs.randomspeaker:
        if cmdargs.dvectordict is None:
            utils.print_error("if randomspeaker is used dvectordict has to be passed")
    if cmdargs.evensplit:
        if cmdargs.augment != 0:
            utils.print_error("When using evensplit, augment has to be 0")
        if cmdargs.maxlen is None:
            utils.print_error("When using evensplit, maxlen has to be set")
        if cmdargs.variableL is not None:
            utils.print_error("When using evensplit, variableL cannot be used")
    # setup output directory and cache commands
    utils.check_output_dir(cmdargs.outdir, True)
    utils.cache_command(sys.argv, cmdargs.outdir)
    utils.change_dir(cmdargs.outdir)
    return cmdargs


def augment_single_meeting(args, basename, meeting_name, seg_list,
                           dvectors, _filenames, _meetings_out, _idx):
    """
        Performs data augmentation on single meeting:
        1. Random shifts of possibly variable length
        2. input randomisation using single or two-level dictionary (meeting,spk)
        If args.augment==0 then meeting is only split and not augmented, used for evaluation
    """
    # NB: filter encompassed already performed in data_loading.py

    # load data and process
    all_spk, all_mat = [], []
    meeting_len = 0
    for segment in seg_list:
        cur_spk = segment[2]
        cur_mat = kaldiio.load_mat(segment[0])
        # l2 norm before average
        cur_mat = cur_mat / np.linalg.norm(cur_mat, axis=1, keepdims=True)
        # average
        cur_mat = np.mean(cur_mat, axis=0, keepdims=True)
        # l2 norm after average
        cur_mat = cur_mat / np.linalg.norm(cur_mat, axis=1, keepdims=True)
        all_spk.append(cur_spk)
        all_mat.append(cur_mat)
        meeting_len += 1
    # concatenate the segment level embeddings
    all_mat = np.concatenate(all_mat, axis=0)
    assert all_mat.shape[0] == len(all_spk) == meeting_len

    meetings_ark, meetings_out = {}, {}
    if args.augment >= 1:
        assert (args.maxlen is not None or args.variableL is not None), "Set maxlen or variableL"
        for i in range(args.augment):
            maxlen = get_maxlen(args, meeting_len)
            start_idx = get_startidx(meeting_len, maxlen)
            cur_meeting_name = meeting_name + '-%03d' % i
            cur_meeting_mat = all_mat[start_idx:(start_idx+maxlen)]
            cur_spk = all_spk[start_idx:(start_idx+maxlen)]
            # replace cur_mat with randomly sampled d-vectors
            if dvectors is not None:
                dvec_dict = dvectors
                if args.randomspeaker:
                    # sorting to make code reproducable
                    spk_in_meeting = sorted(list(set(cur_spk)))
                    # 2-level dictionary, random pick a meeting first
                    if meeting_name in dvectors:
                        all_meeting_names = list(dvectors.keys())
                        # sample a meeting with at the least same number
                        # of speakers as current (sub-)meeting
                        sample_count = 0
                        while True:
                            sample_count += 1
                            assert sample_count < MAXLOOPITERATIONS, "possibly an infinite loop"
                            random_meeting_name = np.random.choice(all_meeting_names)
                            if len(dvectors[random_meeting_name].item()) >= len(spk_in_meeting):
                                break
                        dvec_dict = dvectors[random_meeting_name].item()
                        sorted_keys = sorted(dvec_dict.keys())
                        ordered_dvectors = OrderedDict([])
                        for key in sorted_keys:
                            ordered_dvectors[key] = dvec_dict[key]
                        dvec_dict = ordered_dvectors
                    else:
                        assert list(spk_in_meeting)[0] in dvectors, \
                            "only 1-level or 2-lvel dictionary is allowed"
                    all_speakers = list(dvec_dict.keys())
                    new_spk = np.random.choice(all_speakers, len(spk_in_meeting), replace=False)
                    spk_mapping = {orig_spk: rand_spk
                                   for orig_spk, rand_spk in zip(spk_in_meeting, new_spk)}
                    cur_spk = [spk_mapping[orig_spk] for orig_spk in cur_spk]
                else:
                    if isinstance(next(dvectors.values()), dict):
                        dvec_dict = dvectors[meeting_name]
                samples = [np.random.choice(np.arange(dvec_dict[spk].shape[0])) for spk in cur_spk]
                cur_meeting_mat = [dvec_dict[spk][sample]
                                   for spk, sample in zip(cur_spk, samples)]
                cur_meeting_mat = np.array(cur_meeting_mat)
            else:
                assert args.randomspeaker is False, "randomspeaker not without dvector dictionary"
            cur_label = get_label_from_spk(cur_spk)
            # np.sqrt(cur_meeting_mat.shape[1]) does variance normalisation
            if args.varnormalise is True:
                cur_meeting_mat = cur_meeting_mat * np.sqrt(cur_meeting_mat.shape[1])
            meetings_ark[cur_meeting_name] = cur_meeting_mat
            meetings_out[cur_meeting_name] = cur_meeting_mat.shape, cur_label
    else:
        #maxlen = float('inf') if args.maxlen is None else args.maxlen
        assert args.augment == 0, "invalid augment value"
        # poping out matrices until meet maxlen, form sub meeting
        segment_idx = 0
        while all_mat.size > 0:
            maxlen = get_maxlen(args, meeting_len)
            # pop first maxlen elements from all_mat
            cur_meeting_mat = all_mat[0:maxlen]
            all_mat = all_mat[maxlen:]
            # pop first maxlen elements from all_spk
            cur_spk = all_spk[0:maxlen]
            all_spk = all_spk[maxlen:]
            cur_meeting_name = meeting_name + '-%03d' % segment_idx
            cur_label = get_label_from_spk(cur_spk)
            # np.sqrt(cur_meeting_mat.shape[1]) does variance normalisation
            if args.varnormalise is True:
                cur_meeting_mat = cur_meeting_mat *  np.sqrt(cur_meeting_mat.shape[1])
            meetings_ark[cur_meeting_name] = cur_meeting_mat
            meetings_out[cur_meeting_name] = cur_meeting_mat.shape, cur_label
            segment_idx += 1
            assert all_mat.shape[0] == len(all_spk)

    filename = os.path.join(basename, meeting_name)
    ark_path = utils.get_abs_path(filename + '.ark')
    with kaldiio.WriteHelper('ark,scp:%s,%s.scp' % (ark_path, filename)) as writer:
        for key, mat in meetings_ark.items():
            writer(key, mat)
    _filenames[_idx] = filename
    _meetings_out[_idx] = meetings_out


def get_maxlen(args, meetinglength):
    """ based on variableL, maxlen is randomly set. If variableL is None then maxlen=args.maxlen"""
    if args.variableL is not None:
        if args.maxlen is not None:
            maxlen = int(np.random.uniform(args.variableL[0], args.variableL[1]) * min(meetinglength, args.maxlen))-1
        else:
            maxlen = int(np.random.uniform(args.variableL[0], args.variableL[1]) * meetinglength)-1
    elif args.evensplit:
        assert args.maxlen is not None
        assert args.variableL is None
        maxlen = np.ceil(meetinglength / np.ceil(meetinglength / args.maxlen))
    else:
        if args.maxlen is not None:
            maxlen = args.maxlen
        else:
            maxlen = meetinglength
    return maxlen


def get_startidx(meetinglength, maxlen):
    """
        Pick starting point of sub-meeting uniformly random
    """
    # The commented code is used in the old version of the code
    #TAIL = 30
    #assert meetinglength > 30
    #start_idx = np.random.randint(meetinglength - TAIL)
    start_idx = np.random.randint(meetinglength - maxlen)
    return start_idx


def get_label_from_spk(spk_list):
    """
        Returns dictionary mapping from speaker to output label (0,1,2...)
    """
    spk_mapping = {}
    for spk in spk_list:
        if spk not in spk_mapping:
            spk_mapping[spk] = len(spk_mapping)
    return [spk_mapping[spk] for spk in spk_list]
