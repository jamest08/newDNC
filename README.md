# newDNC

An extension of https://github.com/FlorianKrey/DNC with on-the-fly data augmentation, window-level clustering and TDOA/GCC-PHAT inputs.

Training parameters (including batch_size, epochs and accum_grad) can be changed in espnet/egs/ami/dnc1/conf/tuning/train_transformer.yaml

The number of mini-batches per epoch can be changed in espnet/espnet/asr/pytorch_backend/on_the_fly_iterator.py

First do `source venv/bin/activate`

To produce window-level rttms, run `python3 espnet/data_prep/gen_window_rttm.py`,  setting dataset, input segment-level rttm path, frames_per_dvector and desired_window_length (in frames) in the file.  Window-level rttms have been already produced for window lengths of 0.5s, 1s, 1.5s and 2s (see path below). 

Then `cd espnet/egs/ami/dnc1`

Train the model using:
`./run.sh --stage 4 --stop_stage 4 --init-model model.for.initialisation --emb {dvec, wav2vec2} --tdoa {true, false} --gccphat {true, false} --tdoa-aug {true, false} --permute-aug {true, false} --tdoa-norm {true, false} --dvec-aug {None, meeting, global} --diac {true, false} --meeting-length 50 --train_emb path/to/train.scp --train_rttm path/to/train.rttm --dev_emb path/to/dev.scp --dev_rttm path/to/dev.rttm --tag tag.for.model`

NB: if TDOA/GCC-PHAT are included, data_aug should be None

Produce evaluation files using `python3 espnet/data_prep/prep_eval_files.py`, setting eval-emb, eval-rttm, meeting_length, emb, tdoa, gccphat and output paths in produce_eval_scp(), write_to_json() and write_to_ark().

Decode the model using:
`./run.sh --stage 5 --decode_json /path/to/eval.json --tag tag.for.model`

For scoring, first run:
`python3 scoring/gen_rttm.py --input-scp scoring/scoring_eval.scp --js-dir ~/newDNC/espnet/egs/ami/dnc1/exp/mdm_train_pytorch_tag.for.model/decode_dev_xxxxx/ --js-num 16 --js-name data --rttm-name evaldnc`

Then run:
`python3 espnet/data_prep/gen_reference.py --submeeting-rttm ~/newDNC/espnet/egs/ami/dnc1/exp/mdm_train_pytorch_tag.for.model/decode_dev_xxxxx/evaldnc.rttm --input-rttm ~/newDNC/data/eval_silence_stripped_reference.rttm`

Finally:
`python3 scoring/score_rttm.py --score-rttm /path/to/scoringdir/evaldnc.rttm --ref-rttm ~/newDNC/espnet/data_prep/reference.rttm --output-scoredir /path/to/scoringdir/evaldnc`


Relative paths to segment-level d-vector files:

data/arks.concat/{train, dev, eval}.scp  
data/rttms.concat/{train, dev, eval}.scp 


Relative paths to window-level d-vector files:

data/arks.meeting.cmn.tdnn/{train, dev, eval}.scp  
data/window_level_rttms/

