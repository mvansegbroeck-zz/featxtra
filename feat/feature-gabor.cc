// feat/feature-gabor.cc

// Copyright 2014  Jimmy & Danny
// July 2014: modified by Maarten Van Segbroeck 

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

#include "feat/feature-gabor.h"
#include "time.h"


namespace kaldi {
  
Gabor::Gabor(const GaborOptions &opts):
opts_(opts),
feature_window_function_(opts.frame_opts),
srfft_(NULL) {
  int32 padded_window_size = opts.frame_opts.PaddedWindowSize();
  if ((padded_window_size & (padded_window_size-1)) == 0)  // Is a power of two
    srfft_ = new SplitRadixRealFft<BaseFloat>(padded_window_size);
}

Gabor::~Gabor() {
  for (std::map<BaseFloat, MelBanks*>::iterator iter = mel_banks_.begin();
  iter != mel_banks_.end();
  ++iter)
    delete iter->second;
  if (srfft_ != NULL)
    delete srfft_;
}


const MelBanks *Gabor::GetMelBanks(BaseFloat vtln_warp) {
  MelBanks *this_mel_banks = NULL;
  std::map<BaseFloat, MelBanks*>::iterator iter = mel_banks_.find(vtln_warp);
  if (iter == mel_banks_.end()) {
    this_mel_banks = new MelBanks(opts_.mel_opts,
    opts_.frame_opts,
    vtln_warp);
    mel_banks_[vtln_warp] = this_mel_banks;
  } else {
    this_mel_banks = iter->second;
  }
  return this_mel_banks;
}



void Gabor::ApplyPadding(Matrix<BaseFloat> *spectrogram,
int32 ro, int32 co, Matrix<BaseFloat> *padded_spec) {
  
  // ro: row offset for padding
  // co: col offset for padding
  
  int32 rows_out = spectrogram->NumRows();
  int32 cols_out = spectrogram->NumCols();
  
  // append padding
  SubMatrix<BaseFloat> spec(padded_spec[0], ro, rows_out, co, cols_out);
  spec.CopyFromMat(*spectrogram);
  
  if (opts_.use_reflective_padding) {
    
    if (ro>0) {
      // top side
      SubMatrix<BaseFloat> top_pad((*padded_spec), 0, ro, co, cols_out);
      SubMatrix<BaseFloat> top_spec(*spectrogram, 0, ro, 0, cols_out);
      
      Matrix<BaseFloat> reversed_top_spec(ro, cols_out);
    
      for (int32 i = 0; i < ro; i++) {
        SubVector<BaseFloat> this_row1(top_spec.Row(i));
        SubVector<BaseFloat> this_row2(reversed_top_spec.Row(ro-(i+1)));
        this_row2.CopyFromVec(this_row1);
      }
      top_pad.CopyFromMat(reversed_top_spec);
  
  
      // bottom side
      SubMatrix<BaseFloat> bot_pad((*padded_spec), rows_out+ro, ro, co, cols_out);
      SubMatrix<BaseFloat> bot_spec(*spectrogram, rows_out-ro, ro, 0, cols_out);
      
      Matrix<BaseFloat> reversed_bot_spec(ro, cols_out);
    
      for (int32 i = 0; i < ro; i++) {
        SubVector<BaseFloat> this_row1(bot_spec.Row(i));
        SubVector<BaseFloat> this_row2(reversed_bot_spec.Row(ro-(i+1)));
        this_row2.CopyFromVec(this_row1);
      }
      bot_pad.CopyFromMat(reversed_bot_spec);
  
    }
  
  
    if (co>0) {
      // left side
      SubMatrix<BaseFloat> left_pad((*padded_spec), 0, rows_out+2*ro, 0, co);
      SubMatrix<BaseFloat> left_spec((*padded_spec), 0, rows_out+2*ro, co, co);
  
      Matrix<BaseFloat> left_specT(rows_out+2*ro, co);
      left_specT.CopyFromMat(left_spec);
      left_specT.Transpose();
  
      Matrix<BaseFloat> reversed_left_spec(co, rows_out+2*ro);
    
      for (int32 i = 0; i < co; i++) {
        SubVector<BaseFloat> this_row1(left_specT.Row(i));
        SubVector<BaseFloat> this_row2(reversed_left_spec.Row(co-(i+1)));
        this_row2.CopyFromVec(this_row1);
      }
      reversed_left_spec.Transpose();      
      left_pad.CopyFromMat(reversed_left_spec);
  
      // right side
      SubMatrix<BaseFloat> right_pad((*padded_spec), 0, rows_out+2*ro, cols_out+co, co);
      SubMatrix<BaseFloat> right_spec((*padded_spec), 0, rows_out+2*ro, cols_out, co);
  
      Matrix<BaseFloat> right_specT(rows_out+2*ro, co);
      right_specT.CopyFromMat(right_spec);
      right_specT.Transpose();
  
      Matrix<BaseFloat> reversed_right_spec(co, rows_out+2*ro);
    
      for (int32 i = 0; i < co; i++) {
        SubVector<BaseFloat> this_row1(right_specT.Row(i));
        SubVector<BaseFloat> this_row2(reversed_right_spec.Row(co-(i+1)));
        this_row2.CopyFromVec(this_row1);
      }
      reversed_right_spec.Transpose();      
      right_pad.CopyFromMat(reversed_right_spec);
  
    }
  
  }
}


void Gabor::RemovePadding(Matrix<BaseFloat> input,
int32 ro,
int32 co,
Matrix<BaseFloat> *output) {
  
  int32 inRows = input.NumRows();
  int32 inCols = input.NumCols();
  int32 outRows = inRows-2*ro;
  int32 outCols = inCols-2*co;
  
  output->Resize(outRows, outCols);
      
  SubMatrix<BaseFloat> out(input, ro, outRows, co, outCols);
  output->CopyFromMat(out);
  
}



void Gabor::GFBCalcAxis(Vector<BaseFloat> omega_max,
Vector<BaseFloat> size_max,
Vector<BaseFloat> nu,
Vector<BaseFloat> distance,
Vector<BaseFloat> *omega_n,
Vector<BaseFloat> *omega_k) {
  // % Calculates the modulation center frequencies iteratively.
  // Initialize Vectors  
  Vector<BaseFloat> omega_min;
  omega_min.Resize(size_max.Dim());
  Vector<BaseFloat> c(distance.Dim());
  // c.Resize(distance.Dim());
  omega_n->Resize(1);
  omega_k->Resize(1);
  
  // % Termination condition for iteration is reaching omega_min, which is
  // % derived from size_max.
  omega_min.CopyFromVec(nu);
  omega_min.DivElements(size_max);
  omega_min.Scale(M_PI);
  
  // % Eq. (2b)
  c.CopyFromVec(distance);
  c.DivElements(nu);
  c.Scale(8.0); 
  
  // % Second factor of Eq. (2a)
  BaseFloat space_n = (1.0 + c(1) / 2) / (1.0 - c(1) / 2); 
  int32 count_n = 1;  
  (*omega_n)(0) = omega_max(1); 
  
  // % Iterate starting with omega_max in spectral dimension
  while ( (*omega_n)(count_n-1) /space_n > omega_min(1) ) {
    omega_n->Resize(omega_n->Dim()+1, kCopyData);
    (*omega_n)(count_n) = omega_max(1) / pow(space_n,count_n);
    count_n++;    
  }
  
  // % Add DC
  omega_n->Resize(omega_n->Dim()+1, kCopyData);
  (*omega_n)(omega_n->Dim()) = 0.0;
  
  Vector<BaseFloat> omega_n_tmp(omega_n->Dim());
  omega_n_tmp.CopyFromVec((*omega_n));
  for ( int32 i = 0; i<omega_n->Dim(); i++ ) {
    (*omega_n)(i) = omega_n_tmp(omega_n->Dim()-(i+1));
  }
    
  // % Second factor of Eq. (2a)  
  BaseFloat space_k = (1 + c(0) / 2) / (1 - c(0) / 2); 
  int32 count_k = 1;
  (*omega_k)(0) = omega_max(0);
  
  // % Iterate starting with omega_max in temporal dimension  
  while ( (*omega_k)(count_k-1) / space_k > omega_min(0) ) {
    omega_k->Resize(omega_k->Dim()+1, kCopyData);
    (*omega_k)(count_k) = omega_max(0) / pow(space_k,count_k);
    count_k++;    
  }    
  
  // % Add DC and negative MFs for spectro-temporal opposite 
  // % filters (upward/downward)
  Vector<BaseFloat> omega_k_tmp(omega_k->Dim());
  omega_k_tmp.CopyFromVec((*omega_k));
  omega_k->Resize(2*(omega_k->Dim())+1);
  int32 j = 0;
  while ( j < omega_k_tmp.Dim() ) {
    (*omega_k)(j) = - omega_k_tmp(j);
    j++;
  }
  (*omega_k)(j) = 0; j++;
  while ( j < 2*(omega_k_tmp.Dim())+1 ) {
    (*omega_k)(j) = omega_k_tmp(2*(omega_k_tmp.Dim())-j);
    j++;
  }

}

void Gabor::ComputeHannWindow(BaseFloat width, Vector<BaseFloat> *window) {
  
  int32 width_i = ceil(width);
  
  BaseFloat x_center = 0.5;
  Vector<BaseFloat> x_values(width_i+1);
  
  x_values(width_i/2-1) = (x_center-1.0/(width+1));
  for( int32 i=0; i<width_i/2-1; i++ )
    x_values(width_i/2-i-2) = x_values(width_i/2-i-1) - 1.0/(width+1);
  x_values(width_i/2) = x_center;
  for( int32 i=width_i/2+1; i<width_i+1; i++)
    x_values(i) = x_values(i-1) + 1.0/(width+1);
    
  window->Resize(width_i+1);
  for ( int32 i=0; i<width_i+1; i++ )
    (*window)(i) = 0.5 * (1.0 - (cos(2 * M_PI * x_values(i))));

}  

void Gabor::ComputeMagnitude(Matrix<BaseFloat> real, 
Matrix<BaseFloat> imag, 
Matrix<BaseFloat> *mag) {
  
  real.ApplyPow(2);
  imag.ApplyPow(2);
  
  real.AddMat(1.0, imag);
  
  real.ApplyPow(0.5);
  
  mag->CopyFromMat(real);
  
}


void Gabor::ComputeGaborFilter(BaseFloat omega_k, BaseFloat omega_n, 
Vector<BaseFloat> nu, Vector<BaseFloat> size_max,
Matrix<BaseFloat> *gfilter_real, Matrix<BaseFloat> *gfilter_imag) {
  // % Generates a gabor filter function with:
  // %  omega_k       spectral mod. freq. in rad
  // %  omega_n       temporal mod. freq. in rad
  // %  nu_k          number of half waves unter the envelope in spectral dim.
  // %  nu_n          number of half waves unter the envelope in temporal dim.
  // %  size_max_k    max. allowed extension in spectral dimension
  // %  size_max_n    max. allowed extension in temporal dimension

  // % Calculate windows width.
  BaseFloat w_n = 2*M_PI / abs(omega_n) * nu(0) / 2;
  BaseFloat w_k = 2*M_PI / abs(omega_k) * nu(1) / 2;
      
  // % If the size exceeds the max. allowed extension in a dimension set the
  // % corresponding mod. freq. to zero.
  if( w_n > size_max(1) ) {
    w_n = size_max(1);
    omega_n = 0.0;
  }
  if( w_k > size_max(0) ) {
    w_k = size_max(0);
    omega_k = 0.0;
  }
  
  // % Separable hanning envelope, cf. Eq. (1c).
  Vector<BaseFloat> env_n;
  Vector<BaseFloat> env_k;
  int32 win_size_k = ceil(w_k);
  int32 win_size_n = ceil(w_n);
  
  ComputeHannWindow(w_n-1, &env_n);
  ComputeHannWindow(w_k-1, &env_k);
  
  Matrix<BaseFloat> envelope(win_size_k, win_size_n, kSetZero);
  
  envelope.AddVecVec(1.0, env_k, env_n);
  
  // % Sinusoid carrier, cf. Eq. (1c).
  int32 n_0 = (win_size_n+1) / 2;
  int32 k_0 = (win_size_k+1) / 2;
  
  BaseFloat sinusoid_r;
  BaseFloat sinusoid_i;
  
  gfilter_real->Resize(win_size_k, win_size_n, kSetZero);
  gfilter_imag->Resize(win_size_k, win_size_n, kSetZero);
  
  gfilter_real->CopyFromMat(envelope);
  
  // % Eq. 1c
  for( int32 n=0; n<win_size_n; n++ ) {
    for( int32 k=0; k<win_size_k; k++ ) {
      ComplexImExp((omega_n*(n+1-n_0) + omega_k*(k+1-k_0)), &sinusoid_r, &sinusoid_i);
      
      ComplexMul(sinusoid_r, sinusoid_i, &(*gfilter_real)(k,n), &(*gfilter_imag)(k,n));
    }
  }
  
  // % Compensate the DC part by subtracting an appropiate part
  // % of the envelope if filter is not the DC filter.
  BaseFloat envelope_mean = envelope.Sum() / win_size_k / win_size_n;  
  BaseFloat gfilter_real_mean = gfilter_real->Sum() / win_size_k / win_size_n;
  BaseFloat gfilter_imag_mean = gfilter_imag->Sum() / win_size_k / win_size_n;
  
  Matrix<BaseFloat> comp_r(win_size_k, win_size_n, kSetZero);
  comp_r.CopyFromMat(envelope);
  comp_r.Scale(-gfilter_real_mean/envelope_mean);
  Matrix<BaseFloat> comp_i(win_size_k, win_size_n, kSetZero);
  comp_i.CopyFromMat(envelope);
  comp_i.Scale(-gfilter_imag_mean/envelope_mean);
  
  if( (omega_n != 0) || (omega_k !=0) ) {
    
    gfilter_real->AddMat(1.0, comp_r);
    gfilter_imag->AddMat(1.0, comp_i);
    
  }
  else {
    
    // Add an imaginary part to DC filter for a fair real/imag comparison.
    gfilter_imag->CopyFromMat((*gfilter_real));
    
  }
  
  // 2D FFT
  Matrix<BaseFloat> gfilter_fft_real(win_size_k, win_size_n);
  Matrix<BaseFloat> gfilter_fft_imag(win_size_k, win_size_n);
  gfilter_fft_real.CopyFromMat((*gfilter_real));
  gfilter_fft_imag.CopyFromMat((*gfilter_imag));
  ComputeComplexFft(&gfilter_fft_real, &gfilter_fft_imag, win_size_k, win_size_n, true);

  Matrix<BaseFloat> gfilter_fft_mag(win_size_k, win_size_n);
  ComputeMagnitude(gfilter_fft_real, gfilter_fft_imag, &gfilter_fft_mag);
  
  // % Normalize filter to have gains <= 1.
  BaseFloat maxFftMag;
  maxFftMag = gfilter_fft_mag.Max();
      
  gfilter_real->Scale(1.0/maxFftMag);
  gfilter_imag->Scale(1.0/maxFftMag);
  
}


void Gabor::FftConv2(Matrix<BaseFloat> in1_real,
Matrix<BaseFloat> in1_imag,
Matrix<BaseFloat> in2_real,
Matrix<BaseFloat> in2_imag,
Matrix<BaseFloat> *out_real,
Matrix<BaseFloat> *out_imag) {
  
  int32 size_y = in1_real.NumRows() + in2_real.NumRows() - 1;
  int32 size_x = in1_real.NumCols() + in2_real.NumCols() - 1;
  int32 fft_size_y = pow(2, ceil(log2(size_y)));
  int32 fft_size_x = pow(2, ceil(log2(size_x)));
  int32 outRows = in1_real.NumRows();
  int32 outCols = in1_real.NumCols();
  int32 y_offset = in2_real.NumRows()/2;
  int32 x_offset = in2_real.NumCols()/2;
    
  size_y=fft_size_y;
  size_x=fft_size_x;
        
  ComputeComplexFftPow2(&in1_real, &in1_imag, size_y, size_x, true);
  ComputeComplexFftPow2(&in2_real, &in2_imag, size_y, size_x, true);
  
  
  for (int32 i=0; i<size_y; i++) {
    for (int32 j=0; j<size_x; j++) {
      ComplexMul(in1_real(i,j),in1_imag(i,j),&in2_real(i,j),&in2_imag(i,j));
    }
  }
  
  Matrix<BaseFloat> out_pad_real(size_y, size_x);
  Matrix<BaseFloat> out_pad_imag(size_y, size_x);
  
  out_pad_real.CopyFromMat(in2_real);
  out_pad_imag.CopyFromMat(in2_imag);
  
  ComputeComplexFftPow2(&out_pad_real, &out_pad_imag, size_y, size_x, false);
  out_pad_real.Scale(1.0/(size_y*size_x));
  out_pad_imag.Scale(1.0/(size_y*size_x));
  
  SubMatrix <BaseFloat> this_out_real(out_pad_real, y_offset, outRows, x_offset, outCols);
  SubMatrix <BaseFloat> this_out_imag(out_pad_real, y_offset, outRows, x_offset, outCols);
  
  out_real->Resize(outRows, outCols);
  out_imag->Resize(outRows, outCols);
  
  out_real->CopyFromMat(this_out_real);
  out_imag->CopyFromMat(this_out_imag);
    
}

void Gabor::ApplyGaborFilter(Matrix<BaseFloat> gfilter_real,
Matrix<BaseFloat> gfilter_imag,
Matrix<BaseFloat> spectrogram,
Matrix<BaseFloat> *gfiltered_spec_real,
Matrix<BaseFloat> *gfiltered_spec_imag) {
  // % Applies the filtering with a 2D Gabor filter to log_mel_spec
  // % This includes the special treatment of filters that do not lie fully
  // % inside the spectrogram
  
  BaseFloat gfilter_min;
  gfilter_min = gfilter_real.Min();
  
  Matrix<BaseFloat> dc_map_real(spectrogram.NumRows(), spectrogram.NumCols(), kSetZero);
  Matrix<BaseFloat> dc_map_imag(spectrogram.NumRows(), spectrogram.NumCols(), kSetZero);
  
  // Create zeros matrix for imaginary part of spectrogram
  Matrix<BaseFloat> spec_imag(spectrogram.NumRows(), spectrogram.NumCols(), kSetZero);
      
  if (gfilter_min < 0){
    // % Compare this code to the compensation for the DC part in the
    // % 'gfilter_gen' function. This is an online version of it removing the
    // % DC part of the filters by subtracting an appropriate part of the
    // % filters' envelope.
    
    Matrix<BaseFloat> gfilter_mag(gfilter_real.NumRows(), gfilter_real.NumCols());
    ComputeMagnitude(gfilter_real, gfilter_imag, &gfilter_mag);
    
    BaseFloat gfilter_mag_sum = gfilter_mag.Sum();
    gfilter_mag.Scale(1.0/gfilter_mag_sum);
    
    Matrix<BaseFloat> gfilter_mag_imag(gfilter_real.NumRows(), gfilter_real.NumCols(), kSetZero);
    
    Matrix<BaseFloat> gfilter_dc_map_real(spectrogram.NumRows(), spectrogram.NumCols(), kSetZero);
    gfilter_dc_map_real.Add(1.0);
    Matrix<BaseFloat> gfilter_dc_map_imag(spectrogram.NumRows(), spectrogram.NumCols(), kSetZero);
    
    FftConv2(gfilter_dc_map_real, gfilter_dc_map_imag, gfilter_real, gfilter_imag, &gfilter_dc_map_real, &gfilter_dc_map_imag);
    
    
    Matrix<BaseFloat> env_dc_map_real(spectrogram.NumRows(), spectrogram.NumCols(), kSetZero);
    env_dc_map_real.Add(1.0);
    Matrix<BaseFloat> env_dc_map_imag(spectrogram.NumRows(), spectrogram.NumCols(), kSetZero);
    
    FftConv2(env_dc_map_real, env_dc_map_imag, gfilter_mag, gfilter_mag_imag, &env_dc_map_real, &env_dc_map_imag);
          
    FftConv2(spectrogram, spec_imag, gfilter_mag, gfilter_mag_imag, &dc_map_real, &dc_map_imag);
    
    dc_map_real.DivElements(env_dc_map_real);
    dc_map_imag.DivElements(env_dc_map_real);
    
    for (int32 i=0; i<dc_map_real.NumRows(); i++) {
      for (int32 j=0; j<dc_map_imag.NumCols(); j++) {
        ComplexMul(gfilter_dc_map_real(i,j), gfilter_dc_map_imag(i,j), &dc_map_real(i,j), &dc_map_imag(i,j));
      }
    }
    
  }
  
  FftConv2(spectrogram, spec_imag, gfilter_real, gfilter_imag, &(*gfiltered_spec_real), &(*gfiltered_spec_imag));
  
  gfiltered_spec_real->AddMat(-1.0, dc_map_real);
  gfiltered_spec_imag->AddMat(-1.0, dc_map_imag);
  
    
}


void Gabor::GFBSelectRep(Matrix<BaseFloat> gfilter_real,
Matrix<BaseFloat> gfilter_imag,
Matrix<BaseFloat> *gfiltered_spec_real,
Matrix<BaseFloat> *gfiltered_spec_imag) {
  // % Selects the center channel by choosing k_offset and those with k_factor
  // % channels distance to it in spectral dimension where k_factor is approx.
  // % 1/4 of the filters extension in the spectral dimension.

  int32 k_factor;    
  int32 k_offset;
  int32 k_chans;
  
  k_factor = ( (gfilter_real.NumRows()/4) > 1 ? (gfilter_real.NumRows()/4) : 1); 
  k_offset = (gfiltered_spec_real->NumRows()/2) % k_factor;
  k_chans = (gfiltered_spec_real->NumRows()) / k_factor;
  
  Matrix<BaseFloat> gfiltered_spec_rep_real(k_chans, gfiltered_spec_real->NumCols());
  Matrix<BaseFloat> gfiltered_spec_rep_imag(k_chans, gfiltered_spec_real->NumCols());
  
  for (int32 k=0; k<k_chans; k++) {
    
    SubVector<BaseFloat> this_gsrr(gfiltered_spec_rep_real.Row(k));
    SubVector<BaseFloat> this_gsr(gfiltered_spec_real->Row(k*k_factor+k_offset));
    SubVector<BaseFloat> this_gsri(gfiltered_spec_rep_imag.Row(k));
    SubVector<BaseFloat> this_gsi(gfiltered_spec_imag->Row(k*k_factor+k_offset));
    
    this_gsrr.CopyFromVec(this_gsr);
    this_gsri.CopyFromVec(this_gsi);
  }
  
  gfiltered_spec_real->Resize(k_chans, gfiltered_spec_real->NumCols());
  gfiltered_spec_imag->Resize(k_chans, gfiltered_spec_imag->NumCols());
  
  gfiltered_spec_real->CopyFromMat(gfiltered_spec_rep_real);
  gfiltered_spec_imag->CopyFromMat(gfiltered_spec_rep_imag);
  
}



void Gabor::Compute(const VectorBase<BaseFloat> &wave,
BaseFloat vtln_warp,
Matrix<BaseFloat> *output,
Vector<BaseFloat> *wave_remainder) {
           
  assert(output != NULL);
  int32 rows_out = NumFrames(wave.Dim(), opts_.frame_opts);
  int32 cols_out = opts_.mel_opts.num_bins;
  int32 ro = opts_.padding_time; // row offset for padding
  int32 co = opts_.padding_freq; // col offset for padding
  Matrix<BaseFloat> spectrogram;
  
  if (rows_out == 0)
    KALDI_ERR << "Gabor::Compute, no frames fit in file (#samples is " << wave.Dim() << ")";
  spectrogram.Resize(rows_out, cols_out);
  if (wave_remainder != NULL)
    ExtractWaveformRemainder(wave, opts_.frame_opts, wave_remainder);
  
  Vector<BaseFloat> window;  // windowed waveform
  Vector<BaseFloat> mel_energies;
           
  for (int32 r = 0; r < rows_out; r++) {  // r is frame index.
    
    ExtractWindow(wave, r, opts_.frame_opts, feature_window_function_, &window, NULL);

    if (srfft_) srfft_->Compute(window.Data(), true);  // Compute FFT using
    // split-radix algorithm.
    else RealFft(&window, true);  // An alternative algorithm that
    // works for non-powers-of-two
    
    // Convert the FFT into a power spectrum
    ComputePowerSpectrum(&window);
    SubVector<BaseFloat> power_spectrum(window, 0, window.Dim()/2);
    power_spectrum.ApplyPow(0.5);
    
    // Integrate with MelFiterbank over power spectrum
    const MelBanks *this_mel_banks = GetMelBanks(vtln_warp);
    this_mel_banks->Compute(power_spectrum, &mel_energies);
    
    if (opts_.use_cubed_root)
      mel_energies.ApplyPow(1.0/3); // apply cubed root
    else
      mel_energies.ApplyLog();  // take the log
    
    // Copy to spectrogram
    SubVector<BaseFloat> this_spec(spectrogram.Row(r));
    this_spec.CopyFromVec(mel_energies);

  }
 
  // additional padding for very short utterances
  int32 ro_add = max(ro - rows_out, 0);
  rows_out = rows_out + ro_add;
  spectrogram.Resize(rows_out, cols_out, kCopyData);

  // Apply reflective padding to spectrogram 
  Matrix<BaseFloat> padded_spec(rows_out+2*ro, cols_out+2*co, kSetZero);

  ApplyPadding(&spectrogram, ro, co, &padded_spec);
    
  // Transpose to match gfilters axes
  padded_spec.Transpose();
  
  
  // Gabor filter stuffs
  Vector<BaseFloat> omega_max(2);
  Vector<BaseFloat> size_max(2);
  Vector<BaseFloat> nu(2);
  Vector<BaseFloat> distance(2);
  Vector<BaseFloat> omega_n;
  Vector<BaseFloat> omega_k;
  
  //% Filter bank settings [spectral temporal]
  omega_max(0) = M_PI/2; omega_max(1) = M_PI/2;   //% radians
  size_max(0) = 3*23.0;  size_max(1) = 40.0;   //% bands, frames
  nu(0) = 3.5;  nu(1) = 3.5;   //% half-waves under envelope
  distance(0) = 0.3; distance(1) = 0.2;   //% controls the spacing of filters
  
  // % Calculate center modulation frequencies.
  GFBCalcAxis(omega_max, size_max, nu, distance, &omega_n, &omega_k);
  
  // % selection of first number of temporal modulation frequencies
  omega_n.Resize(opts_.nb_mod_freq, kCopyData);
      
  int32 currentRowsOut = 0;
  for (int32 n=0; n<omega_n.Dim(); n++) {
    for (int32 k=0; k<omega_k.Dim(); k++) {
      if (!((omega_k(k)<0.0) && (omega_n(n)==0.0))) {
        // % Generate filters for all pairs of spectral and temporal modulation frequencies
        Matrix<BaseFloat> gfilter_real;
        Matrix<BaseFloat> gfilter_imag;
        ComputeGaborFilter(omega_k(k), omega_n(n), nu, size_max, &gfilter_real, &gfilter_imag);
        // %% Filter mel spectrogram with filter bank filters and select representative channels.
        Matrix<BaseFloat> gfiltered_spec_real;
        Matrix<BaseFloat> gfiltered_spec_imag;
        ApplyGaborFilter(gfilter_real, gfilter_imag, padded_spec, &gfiltered_spec_real, &gfiltered_spec_imag);
        GFBSelectRep(gfilter_real, gfilter_imag, &gfiltered_spec_real, &gfiltered_spec_imag);
        output->Resize(currentRowsOut + gfiltered_spec_real.NumRows(), gfiltered_spec_real.NumCols(), kCopyData);
        // Copy to output
        if (opts_.use_real) {
          SubMatrix<BaseFloat> this_gfiltered_spec((*output), currentRowsOut, gfiltered_spec_real.NumRows(), 0, gfiltered_spec_real.NumCols());
          this_gfiltered_spec.CopyFromMat(gfiltered_spec_real);
          
        } else {
          SubMatrix<BaseFloat> this_gfiltered_spec((*output), currentRowsOut, gfiltered_spec_imag.NumRows(), 0, gfiltered_spec_imag.NumCols());
          this_gfiltered_spec.CopyFromMat(gfiltered_spec_imag);
        }
        currentRowsOut = currentRowsOut + gfiltered_spec_real.NumRows();      
     }
    }  
  }
  output->Transpose();
  // remove additional padding for very short utterances
  output->Resize(output->NumRows()-ro_add, output->NumCols(), kCopyData);
  RemovePadding(*output, ro, co, output);

 }
}

  
