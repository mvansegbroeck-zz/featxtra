// transform/featxtra-functions.cc

// Copyright 2014 University of Southern California (author: Maarten Van Segbroeck)

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

  MatrixIndexT dim = feats->NumCols();
  MatrixIndexT num_frames = feats->NumRows();
  Matrix<BaseFloat> featsmvn(*feats);

  // Apply the normalization.
  BaseFloat tmp1, tmp2;
  for (int32 d = 0; d < dim; d++) {
    tmp1 = 0;
    tmp2 = 0;
    for (int32 i = 0; i < num_frames-ar_order; i++) {
      if (i < ar_order) {
        (*feats)(i, d) = 0.01*featsmvn(i, d);  // suppress values
      } else if (i == ar_order) {
        for (int32 k = 0; k < ar_order; k++) {
          tmp1 += (*feats)(i-1-k, d);
          tmp2 += featsmvn(i+k, d);
        }
        tmp2 += featsmvn(i+ar_order, d);
        (*feats)(i, d) = ( tmp1 + tmp2 ) / ( 2*ar_order + 1 );
      } else {
       tmp1 += (*feats)(i-1, d) - (*feats)(i-1-ar_order, d);
       tmp2 += featsmvn(i+ar_order, d) - featsmvn(i-1, d);
       (*feats)(i, d) = ( tmp1 + tmp2 ) / ( 2*ar_order + 1 );
      }
    }
  }
}

void ApplySigmoidScale(BaseFloat sigmoidThr,
               BaseFloat sigmoidSlope,
               MatrixBase<BaseFloat> *feats) {
  MatrixIndexT num_rows = feats->NumRows();
  MatrixIndexT num_cols = feats->NumCols();
  for (MatrixIndexT r = 0; r < num_rows; r++) {
    for (MatrixIndexT c = 0; c < num_cols; c++) {
      (*feats)(r, c) = static_cast<BaseFloat>(1 / (Exp(-1 / sigmoidSlope *
                       (2 * (*feats)(r, c) - sigmoidThr)) + 1));
    }
  }
}

void ApplyLtsv(int ctx_win,
               BaseFloat ltsv_sigmoidSlope,
               BaseFloat ltsv_sigmoidThr,
               const MatrixBase<BaseFloat> *feats,
               Matrix<BaseFloat> *ltsv) {
  KALDI_ASSERT(feats != NULL);
  MatrixIndexT dim = feats->NumCols();
  MatrixIndexT num_frames = feats->NumRows();
  // Resize ctx_win if larger than number of frames
  if (num_frames < ctx_win+1)
    ctx_win = num_frames-1;
  Matrix<BaseFloat> featsin(num_frames+ctx_win, dim);
  SubMatrix<BaseFloat> featsappend(feats->Range(num_frames-ctx_win-1,
    ctx_win, 0, dim));
  featsin.Range(0, num_frames, 0, dim).CopyFromMat(*feats);
  featsin.Range(num_frames, ctx_win, 0, dim).CopyFromMat(featsappend);
  (*ltsv).Resize(num_frames, 1);

  Vector<BaseFloat> moving_context(dim), ltsv_bins(dim), ltsv_bins_log(dim);
  moving_context.CopyFromVec(featsin.Row(0));
  moving_context.Scale(round(ctx_win/2));
  for (int32 k = 0; k < round(ctx_win/2); k++)
    moving_context.AddVec(1.0, featsin.Row(k));

  BaseFloat ltsv_val = 0.0;
  for (int32 k = 0; k < num_frames; k++) {
    if (k < round(ctx_win/2)) {
       moving_context.AddVec(-1.0, featsin.Row(0));
    } else {
       moving_context.AddVec(-1.0, featsin.Row(k-round(ctx_win/2)));
    }
    moving_context.AddVec(1.0, featsin.Row(k+round(ctx_win/2)));

    ltsv_bins.CopyFromVec(featsin.Row(k));
    ltsv_bins.DivElements(moving_context);
    ltsv_bins.Scale(100);
    ltsv_bins_log.CopyFromVec(ltsv_bins);
    ltsv_bins_log.ApplyLog();

    // entropy
    ltsv_bins.MulElements(ltsv_bins_log);
    ltsv_bins.Scale(-1);

    // variance
    ltsv_bins.Add(-ltsv_bins.Sum()/dim);
    ltsv_bins.ApplyPow(2.0);

    // ltsv
    if (k < num_frames - round(ctx_win/2))
      ltsv_val = ltsv_bins.Sum()/dim;
    (*ltsv)(k, 0) = ltsv_val;
  }
  // sigmoid
  ApplySigmoidScale(ltsv_sigmoidThr, ltsv_sigmoidSlope, ltsv);
}

void ApplyColSum(const Matrix<BaseFloat> &data,
                  Vector<BaseFloat> *colsum ) {
  MatrixIndexT num_cols = data.NumCols();
  MatrixIndexT num_rows = data.NumRows();
  colsum->Resize(num_rows);
  for (MatrixIndexT r = 0; r < num_rows; r++) {
    (*colsum)(r) = data.Range(r, 1, 0, num_cols).Sum();
  }
}

void ApplyColMean(const Matrix<BaseFloat> &data,
                   Vector<BaseFloat> *colmean ) {
  MatrixIndexT num_cols = data.NumCols();
  ApplyColSum(data, colmean);
  colmean->Scale(1.0/num_cols);
}

void ApplySort(VectorBase<BaseFloat> *s ) {
  std::sort(s->Data(), s->Data()+s->Dim());
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
    moving_context.Resize(ctx_win);  // reset to zero values
    if (k < ctx_win_half) {
      moving_context.Range(0,
       ctx_win_half+k).CopyFromVec(data_copy.Range(0,
       ctx_win_half+k));  // zero padding
    }
    else if (k >= data_tail_range_start) {
      moving_context.Range(0,
       ctx_win_half+num_singval-k).CopyFromVec(data_copy.Range(k-ctx_win_half,
       ctx_win_half+num_singval-k));  // zero padding
    } else {
      moving_context.CopyFromVec(data_copy.Range(k-ctx_win_half, ctx_win));
    }
    ApplySort(&moving_context);
    (*data)(k) = (is_odd_ctx_win == 0 ? (moving_context(ctx_win_half) +
      moving_context(ctx_win_half-1)) / 2 : moving_context(ctx_win_half));
  }
}

void ComputeComplexFft(Matrix<BaseFloat> *real_data,
                        Matrix<BaseFloat> *imag_data,
                        int32 dim0,
                        int32 dim1,
                        bool forward_fft) {
  // Copy input matrices into matrices of desired dimensionality
  real_data->Resize(dim0, dim1, kCopyData);
  imag_data->Resize(dim0, dim1, kCopyData);

  // Apply first FFT to the matrix rows 
  Matrix<BaseFloat> gfilter_fft(2*dim0, dim1);
  for (MatrixIndexT i = 0 ; i < dim0; i++) {  
    gfilter_fft.Row(i*2).CopyFromVec(real_data->Row(i));
    gfilter_fft.Row(i*2 + 1).CopyFromVec(imag_data->Row(i));
  }
  gfilter_fft.Transpose();
  Vector<BaseFloat> tmp_fft1(2*dim0);
  for (MatrixIndexT i = 0 ; i < dim1; i++) {  
    tmp_fft1.CopyFromVec(gfilter_fft.Row(i));
    ComplexFft(&tmp_fft1, forward_fft);
    gfilter_fft.Row(i).CopyFromVec(tmp_fft1);
  }

  // Transpose : fft(A).' 
  gfilter_fft.Transpose();
  Matrix<BaseFloat> gfilter_fft_imag(dim0, dim1);
  Matrix<BaseFloat> gfilter_fft_real(dim0, dim1);
  for (MatrixIndexT i = 0 ; i < dim0; i++) {  
    gfilter_fft_real.Row(i).CopyFromVec(gfilter_fft.Row(i*2));
    gfilter_fft_imag.Row(i).CopyFromVec(gfilter_fft.Row(i*2 + 1));
  }
  gfilter_fft_imag.Transpose();
  gfilter_fft_real.Transpose();

  // Apply second FFT to the matrix rows : fft(fft(A).')
  gfilter_fft.Resize(2*dim1, dim0);
  for (MatrixIndexT i = 0 ; i < dim1; i++) {  
    gfilter_fft.Row(i*2).CopyFromVec(gfilter_fft_real.Row(i));
    gfilter_fft.Row(i*2 + 1).CopyFromVec(gfilter_fft_imag.Row(i));
  }
  gfilter_fft.Transpose();
  Vector<BaseFloat> tmp_fft2(2*dim1);
  for (MatrixIndexT i = 0 ; i < dim0; i++) {  
    tmp_fft2.CopyFromVec(gfilter_fft.Row(i));
    ComplexFft(&tmp_fft2, forward_fft);
    gfilter_fft.Row(i).CopyFromVec(tmp_fft2);
  }

  // Transpose : fft(fft(A).').'
  gfilter_fft.Transpose();
  for (MatrixIndexT i = 0 ; i < dim1; i++) {  
    gfilter_fft_real.Row(i).CopyFromVec(gfilter_fft.Row(i*2));
    gfilter_fft_imag.Row(i).CopyFromVec(gfilter_fft.Row(i*2 + 1));
  }
  gfilter_fft_imag.Transpose();
  gfilter_fft_real.Transpose();
 
  real_data->CopyFromMat(gfilter_fft_real);
  imag_data->CopyFromMat(gfilter_fft_imag);
}

void ComputeComplexFftPow2(Matrix<BaseFloat> *real_data,
                        Matrix<BaseFloat> *imag_data,
                        int32 dim0,
                        int32 dim1,
                        bool forward_fft) {

  if ( (dim0 & (dim0-1)) != 0 || dim0 <= 1)
    KALDI_ERR << "ComputeComplexFftPow2 called with invalid number of points "
              << dim0;
  if ( (dim1 & (dim1-1)) != 0 || dim1 <= 1)
    KALDI_ERR << "ComputeComplexFftPow2 called with invalid number of points "
              << dim1;

  // Copy input matrices into matrices of desired dimensionality
  real_data->Resize(dim0, dim1, kCopyData);
  imag_data->Resize(dim0, dim1, kCopyData);

  Matrix<BaseFloat> *gfilter_fft_imag=real_data;
  Matrix<BaseFloat> *gfilter_fft_real=imag_data;

  // Apply first FFT to the matrix rows 
  gfilter_fft_real->Transpose();
  gfilter_fft_imag->Transpose();
  SplitRadixComplexFft<BaseFloat> srfft1(dim0);
  Vector<BaseFloat> tmp_fft1_real(dim0);
  Vector<BaseFloat> tmp_fft1_imag(dim0);
  for (MatrixIndexT i = 0 ; i < dim1; i++) {  
      tmp_fft1_real.CopyFromVec(gfilter_fft_real->Row(i));
      tmp_fft1_imag.CopyFromVec(gfilter_fft_imag->Row(i));
      srfft1.Compute(tmp_fft1_real.Data(), tmp_fft1_imag.Data(), forward_fft);
      gfilter_fft_real->Row(i).CopyFromVec(tmp_fft1_real);
      gfilter_fft_imag->Row(i).CopyFromVec(tmp_fft1_imag);
  }

  // Transpose : fft(A).' 
  gfilter_fft_imag->Transpose();
  gfilter_fft_real->Transpose();

  // Apply second FFT to the matrix rows : fft(fft(A).')
  SplitRadixComplexFft<BaseFloat> srfft2(dim1);
  Vector<BaseFloat> tmp_fft2_real(dim1);
  Vector<BaseFloat> tmp_fft2_imag(dim1);
  for (MatrixIndexT i = 0 ; i < dim0; i++) {  
      tmp_fft2_real.CopyFromVec(gfilter_fft_real->Row(i));
      tmp_fft2_imag.CopyFromVec(gfilter_fft_imag->Row(i));
      srfft2.Compute(tmp_fft2_real.Data(), tmp_fft2_imag.Data(), forward_fft);
      gfilter_fft_real->Row(i).CopyFromVec(tmp_fft2_real);
      gfilter_fft_imag->Row(i).CopyFromVec(tmp_fft2_imag);
  }

}


// This function is copied from KALDI (feat/pitch-functions.cc)
inline BaseFloat NccfToPov(BaseFloat n) {
  BaseFloat ndash = fabs(n);
  if (ndash > 1.0) ndash = 1.0;  // just in case it was slightly outside [-1, 1]
  BaseFloat r = -5.2 + 5.4 * exp(7.5 * (ndash - 1.0)) + 4.8 * ndash -
    2.0 * exp(-10.0 * ndash) + 4.2 * exp(20.0 * (ndash - 1.0));
  // r is the approximate log-prob-ratio of voicing, log(p/(1-p)).
  BaseFloat p = 1.0 / (1 + exp(-1.0 * r));
  KALDI_ASSERT(p - p == 0);  // Check for NaN/inf
  return p;
}

void ApplyNccfToPov(Matrix<BaseFloat>* kaldi_pitch_feats) {
  MatrixIndexT num_frames = kaldi_pitch_feats->NumRows();
    for (MatrixIndexT frame = 0; frame < num_frames; ++frame) {
      (*kaldi_pitch_feats)(frame, 0) = NccfToPov((*kaldi_pitch_feats)(frame, 0));
  }
}

}  // namespace kaldi
