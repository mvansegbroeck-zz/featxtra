// featbin/apply-ltsv.cc

// Copyright 2014  University of Southern California (author: Maarten Van Segbroeck)

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
using namespace std;

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Apply LTSV (Long Term Spectral Variability) measure on a "
        "matrix of spectral features\n"
        "Per-utterance by default, or per-speaker if utt2spk option provided\n"
        "Usage: apply-ltsv [options] feats-rspecifier feats-wspecifier\n";

    ParseOptions po(usage);
    std::string utt2spk_rspecifier;
    int32 ar_order = 10;  // ARMA filter tab order
    int32 ctx_win  = 50;  // context window parameter
    BaseFloat ltsv_slope = 0.2;  // sigmoid slope parameter
    BaseFloat ltsv_thr   = 0.5;  // sigmoid threshold parameter
    po.Register("ar-order", &ar_order, "Order of the ARMA filtering [default: 10]");
    po.Register("ctx-win", &ctx_win, "Context window frame size [default: 50]");
    po.Register("ltsv-slope", &ltsv_slope, "Sigmoid slope parameter [default: 0.2]");
    po.Register("ltsv-thr", &ltsv_thr, "Sigmoid threshold parameter [default: 0.5]");

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
      //Matrix<BaseFloat> &feats  = feat_reader.Value();
      Matrix<BaseFloat> feat(feat_reader.Value());
      Matrix<BaseFloat> ltsv;
      ApplyArma(ar_order, &feat);
      ApplyLtsv(ctx_win, ltsv_slope, ltsv_thr, &feat, &ltsv);
      feat_writer.Write(utt, ltsv);
      num_done++;
    }
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


