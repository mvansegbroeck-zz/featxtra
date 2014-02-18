// featbin/apply-arma.cc

// Copyright 2009-2011  Microsoft Corporation

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"
#include "transform/featxtra-functions.h"
#include <iostream>
using namespace std;


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Apply ARMA (AutoRegressive-moving-average) normalization to a matrix of features using M-tap FIR\n"
        "Per-utterance by default, or per-speaker if utt2spk option provided\n"
        "Usage: apply-arma [options] feats-rspecifier feats-wspecifier\n";

    ParseOptions po(usage);
    std::string utt2spk_rspecifier;
    int ar_order = 2;
    po.Register("ar-order", &ar_order, "Order of the autoregressive model [default: 2]");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    kaldi::int32 num_done = 0;
    
    std::string feat_rspecifier = po.GetArg(1);
    std::string feat_wspecifier = po.GetArg(2);

    SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
    BaseFloatMatrixWriter feat_writer(feat_wspecifier);
    
    if (utt2spk_rspecifier != "")
      KALDI_ERR << "--utt2spk option not compatible with rxfilename as input "
                 << "(did you forget ark:?)";

    for (;!feat_reader.Done(); feat_reader.Next()) {
      std::string utt = feat_reader.Key();
      Matrix<BaseFloat> feat(feat_reader.Value());
      ApplyArma(ar_order, &feat);
      feat_writer.Write(utt, feat);
      num_done++;
    }
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


