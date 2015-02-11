// transform/featxtra-functions.cc

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

#include "transform/featxtra-functions.h"
#include <vector>
using std::vector;

namespace kaldi {

void ApplyArma(int ar_order,
               MatrixBase<BaseFloat> *feats) {
  KALDI_ASSERT(feats != NULL);
  
  int32 dim = feats->NumCols();
  int32 num_frames = feats->NumRows();
  Matrix<BaseFloat> featsmvn(*feats);

  // Apply the normalization.
  BaseFloat tmp1, tmp2;
  for (int32 d = 0; d < dim; d++) {
    tmp1=0; tmp2=0; 
    for (int32 i = 0; i < num_frames-ar_order; i++) {
      if (i < ar_order) { 
        (*feats)(i,d) = 0.01*featsmvn(i,d) ; // suppress values 
      }
      else if (i == ar_order) { 
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

void ApplySigmoidScale(BaseFloat sigmoidThr, 
               BaseFloat sigmoidSlope,
               MatrixBase<BaseFloat> *feat) {
  feat->Scale(2);
  feat->Add(-sigmoidThr);
  feat->Scale(-1/sigmoidSlope);
  feat->ApplyExp();
  feat->Add(1);
  feat->InvertElements();
}

void ApplyLtsv(int ar_order, 
               int ctx_win, 
               BaseFloat ltsv_sigmoidSlope,
               BaseFloat ltsv_sigmoidThr, 
               MatrixBase<BaseFloat> *feats,
               Matrix<BaseFloat> *ltsv) {

  KALDI_ASSERT(feats != NULL);
  int32 dim = feats->NumCols();
  int32 num_frames = feats->NumRows();
  Matrix<BaseFloat> featsin(*feats);
  featsin.Resize(num_frames+ctx_win, dim, kCopyData);
  Matrix<BaseFloat> featsappend(feats->Range(num_frames-ctx_win-1, ctx_win, 0, dim-1));
  featsin.Range(num_frames, ctx_win, 0, dim-1).CopyFromMat(featsappend);
  (*ltsv).Resize(num_frames, 1);

  Vector<BaseFloat> moving_context(dim), featsnorm(dim), featsnormlog(dim), featsentropy(dim), featsvar(dim);
  moving_context.CopyFromVec(featsin.Row(0));
  moving_context.Scale(round(ctx_win/2));
  for (int32 k = 0; k < round(ctx_win/2); k++)
    moving_context.AddVec(1.0, featsin.Row(k));
    
  BaseFloat ltsv_val;
  for (int32 k = 0; k < num_frames; k++) {
    if (k < round(ctx_win/2)) { 
       moving_context.AddVec(-1.0, featsin.Row(0));
    }
    else {
       moving_context.AddVec(-1.0, featsin.Row(k-round(ctx_win/2)));
    }
    moving_context.AddVec(1.0, featsin.Row(k+round(ctx_win/2)));

    featsnorm.CopyFromVec(featsin.Row(k));
    featsnorm.DivElements(moving_context);
    featsnorm.Scale(100);
    featsnormlog.CopyFromVec(featsnorm);
    featsnormlog.ApplyLog();
    // entropy
    featsentropy.CopyFromVec(featsnorm);
    featsentropy.MulElements(featsnormlog);
    featsentropy.Scale(-1);
    // var
    featsvar.CopyFromVec(featsentropy);
    featsvar.Add(-featsentropy.Sum()/dim);
    featsvar.ApplyPow(2.0);
    // ltsv
    if (k < num_frames - round(ctx_win/2)) 
      ltsv_val = featsvar.Sum()/dim;
    (*ltsv)(k, 0) = ltsv_val;
  }
  // sigmoid 
  ApplySigmoidScale(ltsv_sigmoidThr, ltsv_sigmoidSlope, ltsv);
}

void ApplyColSum( Matrix<BaseFloat> *data,
                  Vector<BaseFloat> *colsum ) {

  MatrixIndexT num_cols = data->NumCols();
  MatrixIndexT num_rows = data->NumRows();
  colsum->Resize(num_rows);
  Matrix<BaseFloat> data_copy(*data);
  data_copy.Transpose();
  for (MatrixIndexT d = 0; d < num_cols; d++) {
    colsum->AddVec(1,data_copy.Row(d));
  }
} 

void ApplyColMean( Matrix<BaseFloat> *data,
                   Vector<BaseFloat> *colmean ) {

  MatrixIndexT num_cols = data->NumCols();
  ApplyColSum(data, colmean);
  colmean->Scale(1.0/num_cols);
} 

void ApplySort( VectorBase<BaseFloat> *s ) { 

  MatrixIndexT num_singval = s->Dim();
  std::vector<std::pair<BaseFloat, MatrixIndexT> > vec(num_singval);
  for (MatrixIndexT d = 0; d < num_singval; d++) {
    BaseFloat val = (*s)(d);
    vec[d] = std::pair<BaseFloat, MatrixIndexT>(val, d);
  }
  std::sort(vec.begin(), vec.end());
  Vector<BaseFloat> s_copy(*s);
  for (MatrixIndexT d = 0; d < num_singval; d++)
    (*s)(d) = s_copy(vec[d].second);
}

void ApplyMedianfiltering(int ctx_win,   
                          VectorBase<BaseFloat> *data ) { 

  MatrixIndexT num_singval = data->Dim();
  Vector<BaseFloat> moving_context;
  Vector<BaseFloat> data_copy(*data);
  int ctx_win_half = ctx_win / 2; // integer division
  int is_odd_ctx_win = ctx_win % 2;
  int data_tail_range_start = num_singval-ctx_win_half+(1-is_odd_ctx_win);
  for (int32 k = 0; k < num_singval; k++) {
    moving_context.Resize(ctx_win); // reset to zero values
    if (k < ctx_win_half) { 
      moving_context.Range(0,ctx_win_half+k).CopyFromVec(data_copy.Range(0, ctx_win_half+k));  // zero padding
    }
    else if (k >= data_tail_range_start) { 
      moving_context.Range(0,ctx_win_half+num_singval-k).CopyFromVec(data_copy.Range(k-ctx_win_half, ctx_win_half+num_singval-k)); // zero padding
    }
    else {
      moving_context.CopyFromVec(data_copy.Range(k-ctx_win_half, ctx_win));
    }
    ApplySort(&moving_context);
    (*data)(k) = (is_odd_ctx_win == 0 ? (moving_context(ctx_win_half) + moving_context(ctx_win_half-1)) / 2 : moving_context(ctx_win_half));
  }
}

}  // namespace kaldi
