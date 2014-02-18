// featbin/extract-dims.cc

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


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Extract a dimension range from features \n"
        "Usage: extract-dims [options] in-rspecifier out-wspecifier\n";

    ParseOptions po(usage);
    
    int32 start = 0;
    int32 end = 0;
    po.Register("start", &start, "If nonnegative, define start or range.");
    po.Register("end", &end, "If nonnegative, define end or range.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string rspecifier = po.GetArg(1);
    std::string wspecifier = po.GetArg(2);

    KALDI_ASSERT(start > 0 || start >= end || end > 0);
    
    BaseFloatMatrixWriter kaldi_writer(wspecifier);
    SequentialBaseFloatMatrixReader kaldi_reader(rspecifier);
    int32 k = 0;
    for (; !kaldi_reader.Done() ; kaldi_reader.Next(), k++) {
        std::string utt = kaldi_reader.Key();
        Matrix<BaseFloat> feats(kaldi_reader.Value());

        KALDI_ASSERT(start <= feats.NumCols() || end <= feats.NumCols());

        Matrix<BaseFloat> to_write(feats.ColRange(start-1, (end-start)+1));
        kaldi_writer.Write(kaldi_reader.Key(), to_write );
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


