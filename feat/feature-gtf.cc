// feat/feature-gtf.cc

// Copyright 2009-2011  Karel Vesely;  Petr Motlicek

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


#include "feat/feature-gtf.h"
#include "feat/feature-window.h"

namespace kaldi {

Gtf::Gtf(const GtfOptions &opts):
    opts_(opts),
    feature_window_function_(opts.frame_opts),
    srfft_(NULL) {
  int num_bins = opts.num_bins;
  int num_ceps = opts.num_ceps;
  Matrix<BaseFloat> dct_matrix(num_bins, num_bins);
  ComputeDctMatrix(&dct_matrix);
  // Note that we include zeroth dct in either case.  If using the
  // energy we replace this with the energy.  This means a different
  // ordering of features than HTK.
  SubMatrix<BaseFloat> dct_rows(dct_matrix, 0, num_ceps, 0, num_bins);
  dct_matrix_.Resize(num_ceps, num_bins);
  dct_matrix_.CopyFromMat(dct_rows);  // subset of rows.

  ComputeGammatoneMatrix(&gammatone_matrix_);
  //KALDI_WARN << gammatone_matrix_ ;

  if (opts.energy_floor != 0.0)
    log_energy_floor_ = log(opts.energy_floor);

  int32 padded_window_size = opts.frame_opts.PaddedWindowSize();
  if ((padded_window_size & (padded_window_size-1)) == 0)  // Is a power of two...
    srfft_ = new SplitRadixRealFft<BaseFloat>(padded_window_size);
}

Gtf::~Gtf() {
  for (std::map<BaseFloat, MelBanks*>::iterator iter = mel_banks_.begin();
      iter != mel_banks_.end();
      ++iter)
    delete iter->second;
  if (srfft_)
    delete srfft_;
}

Vector<BaseFloat> Gtf::GetCosine(Vector<BaseFloat> vector) {
  Vector<BaseFloat> vector_out(vector);
  for (MatrixIndexT i = 0; i < vector.Dim(); i++) vector_out(i) = cos(vector(i));
  return vector_out;
}

void Gtf::ComputeGammatoneMatrix(Matrix<BaseFloat> *gammatone_matrix_) {

  // define variables
  int nfilts = opts_.num_bins;
  int nfft = opts_.frame_opts.PaddedWindowSize();
  int sample_freq = opts_.frame_opts.samp_freq;
  BaseFloat width = 0.5;
  BaseFloat maxfreq = sample_freq/2;
  int minfreq = 50;
  int maxlen = nfft/2;
  
  gammatone_matrix_->Resize(nfilts, maxlen); 

  // fixed constants
  BaseFloat EarQ = 9.26449;
  BaseFloat minBW = 24.7;
  int order = 1;
  Vector<BaseFloat> ucirc_real(maxlen);
  Vector<BaseFloat> ucirc_imag(maxlen);
  for (MatrixIndexT i = 0; i < maxlen; i++) { 
    ucirc_real(i) = cos(M_2PI*i/nfft);
    ucirc_imag(i) = sin(M_2PI*i/nfft);
  }
 
  BaseFloat ERB, B, r, cf, theta, pole_real, pole_imag, T, A11, A12, A13, A14;
  BaseFloat p0r, p0i, p1r, p1i, p2, p3, p4;
  BaseFloat g0r, g0i, g1r, g1i, g2r, g2i, g3r, g3i, g4r, g4i, g5r, g5i;
  BaseFloat gain; 
  for (int32 i = 1; i <= nfilts; i++) {
    
    cf = -(EarQ*minBW) + exp((nfilts+1-i)*(-log(maxfreq + EarQ*minBW) + log(minfreq + EarQ*minBW))/nfilts) * (maxfreq + EarQ*minBW);
    ERB = width*pow(pow(cf/EarQ,order) + pow(minBW,order),1/order);
    B = 1.019*M_2PI*ERB;
    r = exp(-B/sample_freq);
    theta = M_2PI*cf/sample_freq;
    pole_real = r*cos(theta); 
    pole_imag = r*sin(theta); 

    T = 1.0/sample_freq;
    A11 = (2*T*cos(M_2PI*cf*T)/exp(B*T) + 2*sqrt(3+pow(2,1.5))*T*sin(M_2PI*cf*T)/exp(B*T))/(2*T);
    A12 = (2*T*cos(M_2PI*cf*T)/exp(B*T) - 2*sqrt(3+pow(2,1.5))*T*sin(M_2PI*cf*T)/exp(B*T))/(2*T);
    A13 = (2*T*cos(M_2PI*cf*T)/exp(B*T) + 2*sqrt(3-pow(2,1.5))*T*sin(M_2PI*cf*T)/exp(B*T))/(2*T);
    A14 = (2*T*cos(M_2PI*cf*T)/exp(B*T) - 2*sqrt(3-pow(2,1.5))*T*sin(M_2PI*cf*T)/exp(B*T))/(2*T);

    ComplexImExp(static_cast<BaseFloat>(2*cf*M_2PI*T), &p0r, &p0i );
    ComplexImExp(static_cast<BaseFloat>(cf*M_2PI*T), &p1r, &p1i );
    p1r*=2*exp(-B*T)*T; p1i*=2*exp(-B*T)*T;
    p2=cos(cf*M_2PI*T);
    p3=sqrt(3 - pow(2,1.5))* sin(cf*M_2PI*T);
    p4=sqrt(3 + pow(2,1.5))* sin(cf*M_2PI*T);

    g0r = -2*T*p0r+p1r*(p2-p3); g0i = -2*T*p0i+p1i*(p2-p3);
    g1r = -2*T*p0r+p1r*(p2+p3); g1i = -2*T*p0i+p1i*(p2+p3);
    g2r = -2*T*p0r+p1r*(p2-p4); g2i = -2*T*p0i+p1i*(p2-p4);
    g3r = -2*T*p0r+p1r*(p2+p4); g3i = -2*T*p0i+p1i*(p2+p4);
    g4r = -2*pow(exp(2*B*T),-1) - 2*p0r + 2*exp(-B*T) + 2*exp(-B*T)*p0r;   
    g4i = -2*p0i + 2*exp(-B*T)*p0i; 
    ComplexMul(g4r,g4i,&g4r,&g4i);
    ComplexMul(g4r,g4i,&g4r,&g4i);
    g5r = g4r/(pow(g4r,2)+pow(g4i,2));
    g5i = -g4i/(pow(g4r,2)+pow(g4i,2));

    ComplexMul(g1r,g1i,&g0r,&g0i);
    ComplexMul(g2r,g2i,&g0r,&g0i);
    ComplexMul(g3r,g3i,&g0r,&g0i);
    ComplexMul(g5r,g5i,&g0r,&g0i);
   
    gain = sqrt(pow(g0r,2) + pow(g0i,2));
    
    Vector<BaseFloat> gtcol_(maxlen);
    BaseFloat g6r, g6i, g6ic, g7;
    for (MatrixIndexT j = 0; j < maxlen; j++) { 
      g6r = pole_real - ucirc_real(j);
      g6i = pole_imag - ucirc_imag(j);
      g6ic = -pole_imag - ucirc_imag(j);
      ComplexMul(g6r,g6ic,&g6r,&g6i);
      g7 = pow(sqrt(pow(g6r,2) + pow(g6i,2)),-4);
      gtcol_(j) = sqrt(pow(ucirc_real(j)-A11,2) + pow(ucirc_imag(j),2)) * sqrt(pow(ucirc_real(j)-A12,2) + pow(ucirc_imag(j),2)) * 
                      sqrt(pow(ucirc_real(j)-A13,2) + pow(ucirc_imag(j),2)) *  sqrt(pow(ucirc_real(j)-A14,2) + pow(ucirc_imag(j),2)) ;
      gtcol_(j) *= (pow(T,4)/gain) ;
      gtcol_(j) *= g7 ;
    }
    gammatone_matrix_->Row(i-1).CopyFromVec(gtcol_);
  }
}

void Gtf::Compute(const VectorBase<BaseFloat> &wave,
                   BaseFloat vtln_warp,
                   Matrix<BaseFloat> *output,
                   Vector<BaseFloat> *wave_remainder) {
  assert(output != NULL);
  int32 rows_out = NumFrames(wave.Dim(), opts_.frame_opts);
  int32 cols_out = (opts_.apply_dct)? (opts_.use_c0)? opts_.num_ceps : opts_.num_ceps-1 : opts_.num_bins;
  if (rows_out == 0)
    KALDI_ERR << "Gtf::Compute, no frames fit in file (#samples is " << wave.Dim() << ")";
  output->Resize(rows_out, cols_out);
  if (wave_remainder != NULL)
    ExtractWaveformRemainder(wave, opts_.frame_opts, wave_remainder);
  Vector<BaseFloat> window;  // windowed waveform.
  Vector<BaseFloat> mel_energies;
  for (int32 r = 0; r < rows_out; r++) {  // r is frame index..
    BaseFloat log_energy;
    ExtractWindow(0, wave, r, opts_.frame_opts, feature_window_function_, &window,
                  (opts_.use_energy && opts_.raw_energy ? &log_energy : NULL));

    if (opts_.use_energy && !opts_.raw_energy)
      log_energy = VecVec(window, window);

    if (srfft_) srfft_->Compute(window.Data(), true);  // Compute FFT using
    // split-radix algorithm.
    else RealFft(&window, true);  // An alternative algorithm that
    // works for non-powers-of-two.

    // Convert the FFT into a power spectrum.
    ComputePowerSpectrum(&window);
    SubVector<BaseFloat> power_spectrum(window, 0, window.Dim()/2);
    power_spectrum.ApplyPow(0.5);

    SubVector<BaseFloat> this_gtf(output->Row(r));

    // GTF 
    Vector<BaseFloat> gtf(opts_.num_bins);
    gtf.AddMatVec(1.0, gammatone_matrix_, kNoTrans, power_spectrum, 0.0);
    gtf.ApplyPow(1.0/3);
    if (opts_.apply_dct) {
      if (opts_.use_c0) {
        this_gtf.AddMatVec(1.0, dct_matrix_, kNoTrans, gtf, 0.0);
      } else {
        this_gtf.AddMatVec(1.0, dct_matrix_.RowRange(1, dct_matrix_.NumRows()-1), kNoTrans, gtf, 0.0); 
      }
    } else {
      this_gtf.CopyFromVec(gtf);
    }
    
  }
}






} // namespace
