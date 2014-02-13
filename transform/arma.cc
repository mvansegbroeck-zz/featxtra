// transform/arma.cc

// Copyright 2009-2013 Microsoft Corporation
//                     Johns Hopkins University (author: Daniel Povey)

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

#include "transform/arma.h"

namespace kaldi {

void ApplyArma(int ar_order,
               MatrixBase<BaseFloat> *feats) {
  KALDI_ASSERT(feats != NULL);
  
  int32 dim = feats->NumCols() - 1;
  int32 num_frames = feats->NumRows();
  Matrix<BaseFloat> featsmvn(*feats);

  // Apply the normalization.
  BaseFloat tmp1, tmp2;
  for (int32 d = 0; d < dim; d++) {
    tmp1=0; tmp2=0; 
    for (int32 i = ar_order; i < num_frames-ar_order; i++) {
      if (i == ar_order) { 
        for (int32 k = 0; k < ar_order; k++) {
          tmp1 += (*feats)(i-1-k, d);   
          tmp2 += featsmvn(i+k, d);   
        }
        tmp2 += featsmvn(i+ar_order, d);   
        (*feats)(i,d) = ( tmp1 + tmp2 ) / ( 2*ar_order + 1 ); 
      } 
      else 
      { 
       tmp1 += (*feats)(i-1, d) - (*feats)(i-1-ar_order, d);   
       tmp2 += featsmvn(i+ar_order, d) - featsmvn(i-1, d);   
       (*feats)(i,d) = ( tmp1 + tmp2 ) / ( 2*ar_order + 1 ); 
      }
    }
  }
}


}  // namespace kaldi
