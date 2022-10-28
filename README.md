# chirpencoding
First block of the neurobiological lab rotation in 2022/23

## Analysis of chirp responses from 2021

### Data wrangling
- Sort repros: Which can be averaged with which? Avoid pseudoreplication!
  - Which beats where recorded (read EOD multiples in chirp repro metadata)?
  - Do we have multiple recordings for the same beat?

### Analysis functions
Test these on slow beats before applying to all chirp trials
- Compute firing rate as function of time (Gaussian Kernel?)
- Compute beat response by same phase of AM modulation
  - Compute beat peaks by using the zero crossings of the signal with lower EOD$f$ (see beatenvelope.py)
  - Remove all beat peaks that have a chirp
  - Remove all beat peaks at the edges of the recording
  - Draw 6 random beat peaks per trial
  - Compute time window around beat peak
  - Extract spikes in time window
  - Center spike times around beat peak

- Compute chirp triggered **spike** average and compare with permutation and beat response
- Compute chirp triggered **burst** average and compare with permutation and beat response
- Do the two above for a simulated homogenous population and single cells