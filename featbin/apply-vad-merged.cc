// featbin/apply-vad.cc

// Copyright 2014  University of Southern California (author: Maarten Van Segbroeck)

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
        "Apply voice activity detection processing on a frame sequence of speech/non-speech probabilities \n"
        "Usage: apply-vad-merged [options] in-rspecifier out-wspecifier\n";

    ParseOptions po(usage);
    std::string spk2utt_rspecifier;
    int32 prb_pow = 2;
    int32 ctx_win = 40;
    BaseFloat vad_thr = 0.4;
    bool prb_str = false;

    po.Register("prb-pow", &prb_pow, "Power of probability stream prior "
     "to median filtering.");
    po.Register("ctx-win", &ctx_win, "Number of frames of median filtering.");
    po.Register("vad-thr", &vad_thr, "VAD threshold.");
    po.Register("prb-str", &prb_str, "Median filtered probability stream.");
    po.Register("spk2utt", &spk2utt_rspecifier, "rspecifier for speaker to "
     "utterance-list map");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
    int32 num_done = 0, num_err = 0;
    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);

    KALDI_ASSERT(vad_thr > 0 || ctx_win > 0);

    BaseFloatMatrixWriter kaldi_writer(wspecifier);
    SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);

    if (spk2utt_rspecifier != "") {
        SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
        RandomAccessBaseFloatMatrixReader kaldi_reader(rspecifier);

        for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
          std::string spk = spk2utt_reader.Key();
          const std::vector<std::string> &uttlist = spk2utt_reader.Value();
          Matrix<BaseFloat> spkfeats;
          for (size_t i = 0; i < uttlist.size(); i++) {
            std::string utt = uttlist[i];
            if (!kaldi_reader.HasKey(utt)) {
              KALDI_WARN << "Did not find features for utterance " << utt;
              num_err++;
              continue;
            }
            const Matrix<BaseFloat> &uttfeats = kaldi_reader.Value(utt);
            spkfeats.Resize(spkfeats.NumRows() + uttfeats.NumRows(),
             1, kCopyData);
            if (utt.find("_NT") == std::string::npos) {
              spkfeats.Range(spkfeats.NumRows() - uttfeats.NumRows(),
               uttfeats.NumRows(), 0, 1).CopyFromMat(uttfeats);
            }
            num_done++;
          }

          if (spkfeats.NumRows() == 0) {
            KALDI_WARN << "No stats accumulated for speaker " << spk;
          } else {
         
          int32 nb_frames=spkfeats.NumRows();
          KALDI_ASSERT(ctx_win < nb_frames);
          Vector<BaseFloat> vad_out;
          // Mean of the probability streams
          ApplyColMean(spkfeats, &vad_out);
          // Take square of the probability streams
          vad_out.ApplyPow(prb_pow);
          // Median filtering of the resulting VAD probability stream
          ApplyMedianfiltering(ctx_win, &vad_out);

          Matrix<BaseFloat> to_write;
          if (!prb_str) {
             to_write.Resize(nb_frames, 1);
             to_write.CopyColFromVec(vad_out, 0);
             // Apply thresholding
             to_write.Add(-vad_thr);
             to_write.ApplyHeaviside();
          } else {
             to_write.Resize(nb_frames, 2);
             to_write.CopyColFromVec(vad_out, 0);
             Vector<BaseFloat> prob_stream = vad_out;
             // Apply thresholding
             to_write.Add(-vad_thr);
             to_write.ApplyHeaviside();
             // concatenation of the probability stream
             to_write.Resize(nb_frames, 2, kCopyData);
             to_write.Range(0, nb_frames, 1, 1).CopyColFromVec(prob_stream, 0);
          }

          // Write 
          kaldi_writer.Write(spk, to_write);

          KALDI_LOG << "Done accumulating vad labels for speaker " << spk 
                    << " for " << num_done << " segments; " 
                    << num_err << " had errors; "
                    << nb_frames << " frames.";
          }
        }
      }
    //int32 k = 0;
    //for (; !kaldi_reader.Done() ; kaldi_reader.Next(), k++) {
    //    std::string utt = kaldi_reader.Key();
    //    const Matrix<BaseFloat> &feats  = kaldi_reader.Value();

    //    KALDI_ASSERT(ctx_win < feats.NumRows());
    //    Vector<BaseFloat> vad_out;

    //    // Mean of the probability streams
    //    ApplyColMean(feats, &vad_out);
    //    // Median filtering of the resulting VAD probability stream
    //    ApplyMedianfiltering(ctx_win, &vad_out);

    //    Matrix<BaseFloat> to_write(feats.NumRows(), 1);
    //    to_write.CopyColFromVec(vad_out, 0);

    //    // Apply thresholding
    //    to_write.Add(-vad_thr);
    //    to_write.ApplyHeaviside();

    //    kaldi_writer.Write(utt, to_write);
    //}
    //return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


