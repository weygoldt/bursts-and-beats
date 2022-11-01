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

## New analysis plan

1. **Are we recording a pyramidal cell?** Baseline analysis to verify if our cell is a pyramidal cell or P-unit.
   1. Baseline spike rate / burst rate
   2. Isi histogram / KDE
   3. Isi autocorrelation
2. **How does our cell react to electrosensory stimuli?** Plot the cells frequency tuning curve in order to infer if it should theoretically react to our stimuli.
   1. FI-curve (and / or white noise FI-curve)
3. **Does our cell react to a beat or chirp?** Compare our stimulus triggered spiketrains to last years results.
   1. Beat triggered spikes and bursts
   2. Chirp triggered spikes and bursts
   - To do:
   - [ ] Make burst triggered.
   - [ ] Figure out why envelope computation breaks at i = 5.
   - [ ] Check back whether there should be a response because there is none.
   - [ ] Why are spikes clustering at the end?
   - [ ] Find a good metric for variance.
   - [ ] Normalize population vs singlecell activity
   - [ ] Plot nicely
4. **How does our cell react to SAMs?**