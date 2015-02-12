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
