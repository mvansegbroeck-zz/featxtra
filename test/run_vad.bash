#!/bin/bash

threads=8
train_cmd="run.pl"
decode_cmd="run.pl"
source path.sh
export LC_ALL=C

wavin=in/wav

compute-kaldi-pitch-feats --sample-frequency=16000 scp:$wavin.scp ark:- | \
   extract-dims --start=1 --end=1 ark:- ark:out/vprob.ark
compute-gtf-feats --verbose=2 --config=conf/gtf2.conf scp:$wavin.scp ark:- | \
   apply-arma --ar_order=5 ark:- ark:- | \
      apply-ltsv --ar_order=2 ark:- ark:out/ltsv.ark
paste-feats ark:out/ltsv.ark ark:out/vprob.ark ark:- | \
   apply-vad --ctx-win=40 --vad-thr=0.2 ark:- ark:out/vad.ark
