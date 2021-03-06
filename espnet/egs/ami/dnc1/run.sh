#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Modified by Qiujia Li for DNC

all_args="$@"
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=4       # start from -1 if you need to start from data download
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1

num_speaker=4
init_model=""
train_sample=1 # THIS USED TO BE 0.2
rotate=false  # HAVE CHANGED THIS FROM TRUE TO FALSE.  AUG FUNCTIONS CAN CHOOSE DIAC ON-THE-FLY

# feature configuration
train_json=
dev_json=
decode_json=

emb=
tdoa=
gccphat=
tdoa_aug=
permute_aug=
tdoa_norm=
diac=
dvec_aug=
meeting_length=
train_rttm=
valid_rttm=
train_emb=
valid_emb=

train_config=conf/tuning/train_transformer.yaml
decode_config=conf/decode.yaml

# rnnlm related
use_wordlm=false    # false means to train/use a character LM
use_lm=false

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
decode_set=dev

# You may set 'mic' to:
#  ihm [individual headset mic- the default which gives best results]
#  sdm1 [single distant microphone- the current script allows you only to select
#        the 1st of 8 microphones]
#  mdm8 [multiple distant microphones-- currently we only support averaging over
#       the 8 source microphones].
# ... by calling this script as, for example,
# ./run.sh --mic sdm1
# ./run.sh --mic mdm8
mic=mdm

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

base_mic=${mic//[0-9]/} # sdm, ihm or mdm
nmics=${mic//[a-z]/} # e.g. 8 for mdm8.

train_set=${mic}_train
train_dev=${mic}_dev
recog_set="${mic}_${decode_set}"

feat_tr_dir=${dumpdir}/${train_set}/dvector; mkdir -p ${feat_tr_dir}
feat_dt_dir=${dumpdir}/${train_dev}/dvector; mkdir -p ${feat_dt_dir}

dict=data/lang_1char/${train_set}_units.txt
mkdir -p data/lang_1char
if [ ! -f ${dict} ]; then
    for i in `seq 0 $(expr ${num_speaker} - 1)`
    do
        echo "$i $i" >> ${dict}
    done
fi


if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=exp/${expname}
mkdir -p ${expdir}
echo "-------------------------------
run.sh ${all_args}
-------------------------------" >> ${expdir}/run.cmds


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "stage 4: Network Training"

    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --init-model ${init_model} \
        --train-sample-rate ${train_sample} \
        --rotate ${rotate} \
        --seed ${seed} \
        --emb ${emb} \
        --tdoa ${tdoa} \
        --gccphat ${gccphat} \
        --tdoa-aug ${tdoa_aug} \
        --permute-aug ${permute_aug} \
        --tdoa-norm ${tdoa_norm} \
        --diac ${diac} \
        --dvec-aug ${dvec_aug} \
        --meeting-length ${meeting_length} \
        --train-rttm ${train_rttm} \
        --valid-rttm ${valid_rttm} \
        --train-emb ${train_emb} \
        --valid-emb ${valid_emb}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=16

    pids=() # initialize pids
    (
        decode_dir=decode_${recog_set}_$(basename ${decode_config%.*})
        echo ${decode_dir}
        feat_recog_dir=${dumpdir}/${recog_set}/dvector

        # split data
        cp ${decode_json} ${feat_recog_dir}/data.json
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}
    ) &
    pids+=($!) # store background pids
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi

