// featbin/apply-vad.cc

// Copyright 2009-2011  Microsoft Corporation

// See ../../COPYING for clarification regarding multiple authors
//
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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Extract a dimension range from features \n"
        "Usage: apply-vad [options] in-rspecifier out-wspecifier\n";

    ParseOptions po(usage);
    
    int32 ctx_win = 20;
    BaseFloat vad_thr = 0.2;
    po.Register("ctx-win", &ctx_win, "Define number of frames of median filtering.");
    po.Register("vad-thr", &vad_thr, "Define VAD vad-threshold.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);

    KALDI_ASSERT(vad_thr > 0 && ctx_win > 0);
    
    BaseFloatMatrixWriter kaldi_writer(wspecifier);
    SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
    int32 k = 0;
    for (; !kaldi_reader.Done() ; kaldi_reader.Next(), k++) {
        std::string utt = kaldi_reader.Key();
        Matrix<BaseFloat> feats(kaldi_reader.Value());

        KALDI_ASSERT(ctx_win < feats.NumRows());
        Vector<BaseFloat> vad_out;
   
        // Mean of the probability streams
        ApplyColMean(feats, &vad_out);
        // Median filtering of the resulting VAD probability stream
        ApplyMedianfiltering(ctx_win, &vad_out); 

        Matrix<BaseFloat> to_write(feats.NumRows(), 1);
        to_write.CopyColFromVec(vad_out, 0);

        // Apply thresholding 
        to_write.Add(-vad_thr);
        to_write.ApplyHeaviside(); 

        kaldi_writer.Write(kaldi_reader.Key(), to_write );
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


