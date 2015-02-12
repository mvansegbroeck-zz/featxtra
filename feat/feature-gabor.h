// feat/feature-gabor.h

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

#ifndef KALDI_FEAT_FEATURE_GABOR_H_
#define KALDI_FEAT_FEATURE_GABOR_H_

#include <string>

#include "feat/feature-functions.h"
#include "transform/featxtra-functions.h"

#include <iostream>
using namespace std;

namespace kaldi {
	/// @addtogroup  feat FeatureExtraction
	/// @{

	struct GaborOptions {
		FrameExtractionOptions frame_opts;
	    MelBanksOptions mel_opts;
	    BaseFloat energy_floor;
		bool use_cubed_root;
		int32 padding_time;
		int32 padding_freq;
		bool use_reflective_padding;
		int32 nb_mod_freq;
		bool use_real;
	  
		GaborOptions():   mel_opts(24),	// e.g. 24: num spectrogram frequency bins.
						  energy_floor(0.0),  // not in log scale: a small value e.g. 1.0e-10
						  use_cubed_root(false),
						  padding_time(50),
						  padding_freq(0),
						  use_reflective_padding(true), // otherwise use zero padding
						  nb_mod_freq(2),
						  use_real(true) {}
		
		void Register(ParseOptions *po) {
			frame_opts.Register(po);
			mel_opts.Register(po);
			po->Register("energy-floor", &energy_floor,
					     "Floor on energy (absolute, not relative) in FBANK computation");
			po->Register("use-cubed-root", &use_cubed_root,
					     "If true, produce cube-root-filterbank (else produce log).");
			po->Register("padding-time", &padding_time,
					     "Number of frames for padding of spectrogram.");
			po->Register("padding-freq", &padding_freq,
					     "Number of frequency bins for padding of spectrogram.");
			po->Register("nb-mod-freq", &nb_mod_freq,
					     "Number of modulation frequencies for Gabor filter-bank.");
			po->Register("use-reflective-padding", &use_reflective_padding,
					     "Use reflective padding of spectrogram (else use zero padding).");
			po->Register("use-real", &use_real,
						 "Use real output of GFB filtered spectrogram (else use imaginary).");
		}	                   	 
	};		

	
	class MelBanks;			
  
  
	class Gabor {
	public:
		Gabor(const GaborOptions &opts);
		~Gabor();

		//int32 Dim() { return opts_.mel_opts; }
 
 
		void Compute(const VectorBase<BaseFloat> &wave,
		BaseFloat vtln_warp,
		Matrix<BaseFloat> *output,
		Vector<BaseFloat> *wave_remainder = NULL);
		
		void ApplyPadding(Matrix<BaseFloat> *spectrogram,
		int32 ro,
		int32 co,
		Matrix<BaseFloat> *padded_spec);
		
		void RemovePadding(Matrix<BaseFloat> input,
		int32 ro,
		int32 co,
		Matrix<BaseFloat> *output);
		
		void GFBCalcAxis(Vector<BaseFloat> omega_max,
		Vector<BaseFloat> size_max,
		Vector<BaseFloat> nu,
		Vector<BaseFloat> distance,
		Vector<BaseFloat> *omega_n,
		Vector<BaseFloat> *omega_k);
		
		void ComputeGaborFilter(BaseFloat omega_k,
		BaseFloat omega_n,
		Vector<BaseFloat> nu,
		Vector<BaseFloat> size_max,
		Matrix<BaseFloat> *gfilter_real,
		Matrix<BaseFloat> *gfilter_imag);
		
		void ComputeHannWindow(BaseFloat width,
		Vector<BaseFloat> *window);
		
		void ComputeMagnitude(Matrix<BaseFloat> real,
		Matrix<BaseFloat> imag,
		Matrix<BaseFloat> *mag);
		
		void ApplyGaborFilter(Matrix<BaseFloat> gfilter_real,
		Matrix<BaseFloat> gfilter_imag,
		Matrix<BaseFloat> spectrogram,
		Matrix<BaseFloat> *gfilter_spec_real,
		Matrix<BaseFloat> *gfilter_spec_imag);
		
		void FftConv2(Matrix<BaseFloat> in1_real,
		Matrix<BaseFloat> in1_imag,
		Matrix<BaseFloat> in2_real,
		Matrix<BaseFloat> in2_imag,
		Matrix<BaseFloat> *out_real,
		Matrix<BaseFloat> *out_imag);
			
		void GFBSelectRep(Matrix<BaseFloat> gfilter_real,
		Matrix<BaseFloat> gfilter_imag,
		Matrix<BaseFloat> *gfiltered_spec_real,
		Matrix<BaseFloat> *gfiltered_spec_imag);
  
  
	private:
		const MelBanks *GetMelBanks(BaseFloat vtln_warp);
		GaborOptions opts_;
		std::map<BaseFloat, MelBanks*> mel_banks_;  // BaseFloat is VTLN coefficient.
		FeatureWindowFunction feature_window_function_;
		SplitRadixRealFft<BaseFloat> *srfft_;
		KALDI_DISALLOW_COPY_AND_ASSIGN(Gabor);
	};	 
	
}

#endif  // KALDI_FEAT_FEATURE_GABOR_H_
