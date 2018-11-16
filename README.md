# Featxtra
The featxtra toolbox lists a set of front-end tools and Signal Processing functions for [Kaldi](http://kaldi.sourceforge.net).

The toolbox includes:
* feature extraction for speech signals
  - Gammatone Frequency Representation, Gammatone Frequency Cepstral Coefficients
  - Gabor Features
  - DCT features (apply on ltsv, voicing stream)
* Voice Activity Detection 
* additional signal filter operations

## List of Functions

### featbin
* featbin/compute-gtf-feats
  - extract Gammatone Frequency Representation (GTF)
  - extract Gammatone Frequency Cepstral Coefficients (GFCC)
* featbin/compute-gabor-feats
  - extract Gabor Features (GBF)
* featbin/compute-dct-feats
  - extract DCT Features from time domain signal
* featbin/apply-arma
  - apply Auto-Regressive Moving Average (ARMA) filtering on a spectral representation (e.g GTF )
* featbin/apply-ltsv
  - apply Long-Term Spectral Variabilityi (LTSV) stream on spectral representation (e.g GTF )
* featbin/apply-vad
  - apply Voice Activity Detection (VAD) using voicing and LTSV probability
* featbin/extract-dims
  - extract specified dimension range out of a feature matrix

### feat
* feat/feature-gtf
  - GTF and GFCC feature implementation code

### transform
* transform/featxtra-functions
  - additional signal processing functions
_____

## Instructions to run
1. Modify the `Makefile` in directories `feat`, `featbin`, and `transform` as follow:
  - feat/Makefile should include `feature-dctf.o feature-gtf.o feature-gabor.o` under `OBJFILES` variable.
  - featbin/Makefile should include `compute-dctf-feats compute-gtf-feats compute-gabor-feats apply-arma apply-ltsv apply-nccf-to-pov apply-vad-merged apply-vad extract-dims` under the `BINFILES` variable.
  - transform/Makefile should include `featxtra-functions.o` under the `OBJFILES` variable.
2. Add the following code in the `src/feat/feature-window.cc` file
```c
void ExtractWaveformRemainder(const VectorBase<BaseFloat> &wave,
                              const FrameExtractionOptions &opts,
                              Vector<BaseFloat> *wave_remainder) {
  int32 frame_shift = opts.WindowShift();
  int32 num_frames = NumFrames(wave.Dim(), opts);
  // offset is the amount at the start that has been extracted.
  int32 offset = num_frames * frame_shift;
  KALDI_ASSERT(wave_remainder != NULL);
  int32 remaining_len = wave.Dim() - offset;
  wave_remainder->Resize(remaining_len);
  KALDI_ASSERT(remaining_len >= 0);
  if (remaining_len > 0)
    wave_remainder->CopyFromVec(SubVector<BaseFloat>(wave, offset, remaining_len));
}
```
3. Add the following code in the `src/featbin/feature-window.h` file
```c
// ExtractWaveformRemainder is useful if the waveform is coming in segments.
// It extracts the bit of the waveform at the end of this block that you
// would have to append the next bit of waveform to, if you wanted to have
// the same effect as everything being in one big block.
void ExtractWaveformRemainder(const VectorBase<BaseFloat> &wave,
                              const FrameExtractionOptions &opts,
                              Vector<BaseFloat> *wave_remainder);
```

Steps 2 and 3 are done because of [this](https://github.com/kaldi-asr/kaldi/commit/1180e467c8ca273c7704199bd27cb734509e931e) commit in the kaldi-asr project.

After these steps just run the `make` command again in the src directory to finally integrate these in your kaldi project.
