// featbin/compute-dctf-feats.cc

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
#include "feat/feature-dctf.h"
#include "transform/featxtra-functions.h"
using namespace std;

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Compute DCT transform by windowing a one dimensional time domain signal \n"
        "Per-utterance by default, or per-speaker if utt2spk option provided\n"
        "Usage: compute-dctf-feats [options] feats-rspecifier feats-wspecifier\n";

    DctfOptions dctf_opts;
    ParseOptions po(usage);
    std::string utt2spk_rspecifier;
    int32 cep_order = 5;  // ARMA filter tab order
    int32 ctx_win  = 10;  // context window parameter

    po.Register("cep-order", &cep_order, "Order of the Cepstral filtering [default: 5]");
    po.Register("ctx-win", &ctx_win, "Context window frame size [default: 30]");

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

    // DCT 
    dctf_opts.num_ceps= cep_order;
    dctf_opts.num_bins= ctx_win;
    dctf_opts.frame_opts.samp_freq=1;
    dctf_opts.frame_opts.frame_length_ms=ctx_win*1000;
    dctf_opts.frame_opts.frame_shift_ms=1000;
    dctf_opts.frame_opts.round_to_power_of_two=false;
    dctf_opts.frame_opts.window_type="rectangular";
    dctf_opts.frame_opts.remove_dc_offset=0.0;
    dctf_opts.frame_opts.dither=0.0;
    dctf_opts.frame_opts.preemph_coeff=0.0;
    Dctf dctf(dctf_opts);

    for (;!feat_reader.Done(); feat_reader.Next()) {
      std::string utt = feat_reader.Key();
      const Matrix<BaseFloat> &feats  = feat_reader.Value();
      if (feats.NumRows() > ctx_win  ) {
        Matrix<BaseFloat> to_write(feats.NumRows(), feats.NumCols()*cep_order);
        for (size_t i = 0; i < feats.NumCols(); i++) { 
          Vector<BaseFloat> featvec(feats.NumRows()); 
          featvec.CopyColFromMat(feats, i);
          Matrix<BaseFloat> features;
          try {
          dctf.Compute(featvec, &features, NULL);
          } catch (...) {
            KALDI_WARN << "Failed to compute features for utterance "
                       << utt;
            continue;
          }
          to_write.Range(0, features.NumRows(), i*(cep_order-1), cep_order).CopyFromMat(features);
          // Repeat last frame to make output of same length as input
          for (size_t j = 0; j < ctx_win-1; j++) {
            to_write.Row(features.NumRows() + j).Range(i*(cep_order-1), cep_order).CopyFromVec(features.Row(features.NumRows()-1));
          }
        }
        feat_writer.Write(utt, to_write);
        num_done++;
      }
    }
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


