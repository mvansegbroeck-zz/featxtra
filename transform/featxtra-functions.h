// transform/cmvn.h

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


#ifndef KALDI_TRANSFORM_FEATXTRA_FUNCTIONS_H_
#define KALDI_TRANSFORM_FEATXTRA_FUNCTIONS_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "feat/feature-functions.h"

namespace kaldi {

// Apply AutoRegressive Moving Average normalization to a matrix
void ApplyArma(int ar_order,
               MatrixBase<BaseFloat> *feats);
// Apply Sigmoid scaling to a matrix
void ApplySigmoidScale(BaseFloat sig_thr,
               BaseFloat sig_slope,
               MatrixBase<BaseFloat> *feats);
// Apply Long-Term Spectral Variability to a matrix
void ApplyLtsv(int ctx_win,
               BaseFloat ltsv_sigmoidSlope,
               BaseFloat ltsv_sigmoidThr,
               const MatrixBase<BaseFloat> *feats,
               Matrix<BaseFloat> *ltsv);
// Compute sum over matrix columns
void ApplyColSum(const Matrix<BaseFloat> &data,
                  Vector<BaseFloat> *colsum);
// Compute mean over matrix columns
void ApplyColMean(const Matrix<BaseFloat> &data,
                   Vector<BaseFloat> *colmean);
// Sort a vector
void ApplySort(VectorBase<BaseFloat> *s);
// Apply median filtering to a time domain signal (vector)
void ApplyMedianfiltering(int ctx_win,
               VectorBase<BaseFloat> *data);
// Apply 2-dimensional FFT to a matrix of real and imaginary numbers
//  function is implemented as fft(fft(A).').'
void ComputeComplexFft(Matrix<BaseFloat> *real_data,
	Matrix<BaseFloat> *imag_data,
	int32 dim0,
	int32 dim1,
	bool forward_fft);
// Apply 2-dimensional FFT to a matrix of real and imaginary numbers
//  use this if matrix dimensions are a power of 2
void ComputeComplexFftPow2(Matrix<BaseFloat> *real_data,
	Matrix<BaseFloat> *imag_data,
	int32 dim0,
	int32 dim1,
	bool forward_fft);

// Convert NCCF to POV conversion on kaldi pitch feats
// Kaldi pitch feats are 2-dim features (NCCF, pitch in Hz)
void ApplyNccfToPov(Matrix<BaseFloat>* kaldi_pitch_feats);

}  // namespace kaldi

#endif  // KALDI_TRANSFORM_FEATXTRA_FUNCTIONS_H_
