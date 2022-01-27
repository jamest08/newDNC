# newDNC

An extension of https://github.com/FlorianKrey/DNC with on-the-fly data augmentation.

Changes to window-level segmentation and overlapping speakers are in development.

To run, first do `source venv/bin/activate`

Then `cd espnet/egs/ami/dnc1`

`./run.sh --stage 4 --stop_stage 4 --train_scp path/to/train.scp --train_rttm path/to/train.rttm --dev_scp path/to/dev.scp --dev_rttm path/to/dev.rttm --tag tag.for.model`



Relative paths to data files:

Train scp: data/arks.concat/train.scp 

Train rttm: data/rttms.concat/train.rttm

Dev scp: data/arks.concat/dev.scp

Dev rttm: data/rttms.concat/dev.rttm

Eval scp: data/arks.concat/eval.scp

Eval rttm: data/rttms.concat/eval.rttm
