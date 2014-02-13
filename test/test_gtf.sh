#!/bin/bash

train_cmd="run.pl"
decode_cmd="run.pl"
source path.sh
export LC_ALL=C

wavin=in/wav
featout=out/test_gtf

compute-gtf-feats --verbose=2 --config=conf/gtf.conf scp:$wavin.scp ark,scp:$featout.ark,$featout.scp
compute-cmvn-stats --spk2utt=ark:in/spk2utt scp:$featout.scp ark:- | \
  apply-cmvn --norm-vars=true --utt2spk=ark:in/utt2spk ark:- scp:$featout.scp ark:$featout.ark
