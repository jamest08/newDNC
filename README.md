# newDNC

An extension of https://github.com/FlorianKrey/DNC with on-the-fly data augmentation, window-level clustering and TDOA/GCC-PHAT inputs.

To run, first do `source venv/bin/activate`

Then `cd espnet/egs/ami/dnc1`

`./run.sh --stage 4 --stop_stage 4 --init-model model.for.initialisation --emb {dvec, wav2vec2} --tdoa {true, false} --gccphat {true, false} --tdoa-aug {true, false} --permute-aug {true, false} --tdoa-norm {true, false} --dvec-aug {None, meeting, global} --diac {true, false} --meeting-length 50 --train_emb path/to/train.scp --train_rttm path/to/train.rttm --dev_emb path/to/dev.scp --dev_rttm path/to/dev.rttm --tag tag.for.model`

Relative paths to segment-level d-vector files:

Train scp: data/arks.concat/train.scp 
Train rttm: data/rttms.concat/train.rttm
Dev scp: data/arks.concat/dev.scp
Dev rttm: data/rttms.concat/dev.rttm
Eval scp: data/arks.concat/eval.scp
Eval rttm: data/rttms.concat/eval.rttm

Python scripts in espnet/data_prep can be used to produce window-level rttms and files for evaluation.
