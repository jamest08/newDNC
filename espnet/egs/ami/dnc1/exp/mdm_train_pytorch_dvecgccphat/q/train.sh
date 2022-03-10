#!/bin/bash
cd /home/mifs/jhrt2/newDNC/espnet/egs/ami/dnc1
. ./path.sh
( echo '#' Running on `hostname`
  echo '#' Started at `date`
  echo -n '# '; cat <<EOF
asr_train.py --config conf/tuning/train_transformer.yaml --ngpu 1 --backend pytorch --outdir exp/mdm_train_pytorch_dvecgccphat/results --tensorboard-dir tensorboard/mdm_train_pytorch_dvecgccphat --debugmode 1 --dict data/lang_1char/mdm_train_units.txt --debugdir exp/mdm_train_pytorch_dvecgccphat --minibatches 0 --verbose 0 --resume --init-model --train-sample-rate 0.2 --rotate false --seed 1 --dvec true --tdoa false --gccphat true --tdoa-aug false --permute-aug false 
EOF
) >exp/mdm_train_pytorch_dvecgccphat/train.log
time1=`date +"%s"`
 ( asr_train.py --config conf/tuning/train_transformer.yaml --ngpu 1 --backend pytorch --outdir exp/mdm_train_pytorch_dvecgccphat/results --tensorboard-dir tensorboard/mdm_train_pytorch_dvecgccphat --debugmode 1 --dict data/lang_1char/mdm_train_units.txt --debugdir exp/mdm_train_pytorch_dvecgccphat --minibatches 0 --verbose 0 --resume --init-model --train-sample-rate 0.2 --rotate false --seed 1 --dvec true --tdoa false --gccphat true --tdoa-aug false --permute-aug false  ) 2>>exp/mdm_train_pytorch_dvecgccphat/train.log >>exp/mdm_train_pytorch_dvecgccphat/train.log
ret=$?
time2=`date +"%s"`
echo '#' Accounting: time=$(($time2-$time1)) threads=1 >>exp/mdm_train_pytorch_dvecgccphat/train.log
echo '#' Finished at `date` with status $ret >>exp/mdm_train_pytorch_dvecgccphat/train.log
[ $ret -eq 137 ] && exit 100;
touch exp/mdm_train_pytorch_dvecgccphat/q/sync/done.27012
exit $[$ret ? 1 : 0]
## submitted with:
# qsub -v PATH -cwd -S /bin/bash -j y -P nst -o exp/mdm_train_pytorch_dvecgccphat/q/train.log -l qp=cuda-low -l osrel='*' -l gpuclass='*' -l not_host=air209   /home/mifs/jhrt2/newDNC/espnet/egs/ami/dnc1/exp/mdm_train_pytorch_dvecgccphat/q/train.sh >>exp/mdm_train_pytorch_dvecgccphat/q/train.log 2>&1
