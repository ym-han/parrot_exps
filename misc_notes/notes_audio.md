# Notes on audio stuff


## General background

From:
`https://www.youtube.com/c/valeriovelardothesoundofai`  
`https://pudding.cool/2018/02/waveforms/`
Meinard Muller's book


A waveform := a graph that shows a wave's change in displacement over time. 
A waveform's amplitude controls the wave's maximum displacement.

Frequency is a measure of how many times the waveform repeats in a given amount of time. Measured in Hertz: the number of repetitions per second.
* Frequency is similar to "pitch". The faster a wave repeats itself, the higher the pitch of the note.

When a waveform has "side effect" frequencies, we call them harmonics.


Sound power: 
* := the rate at which energy is transferred;
* i.e., the energy per unit time emitted by a sound source in all directions
* Measured in watts

Sound intensity:
* := Sound power per unit area
* Measured in W/m^2

An 'intensity level' is a ratio between two intensity values; this is measured in decibels


Intuitively, the *envelope* of a waveform = a smooth curve outlining its extremes in amplitude.
* In sound synthesis, the envelope of a signal to be generated is often described by a model called *ADSR*, which consists of an attack (A), decay (D), sustain (S), and release (R) phase 
 
What is a *spectrogram*?
* a visual representation of the spectrum of frequencies of a signal as it varies with time
 
Sampling rate:
* The values in a digital signal are based on sampled (i.e., measured) amplitude values of the analog signal. The samples in a digital signal occur at regular time intervals, T_s, called the *sampling period* with units of seconds per sample. 
* The sampling rate, F_s := 1 / T_s, of a digital signal is the
number of samples per second.



# Audio frames

## The intuitive idea, from Valerio Velardo

The duration of a sample --- the sampling period --- is going to be very small, too small for a human to 'apprecite it as a acoustic event'.

Since we want to extract features that are relevant to what we are hearing in the audio, we want to use slices of audio that are longer than a sample as a basis for extracting features. We can think of a frame as being such a 'perceivable audio chunk' --- a slice of the audio signal that's long enough for humans to register it as an acoustic event.

*Frame size*: number of samples that we have in a frame; this is usually a power of 2. (Having a number of samples that's a power of 2 speeds up fast fourier transform; helpful for when we want to get frequency features)

Duration of a frame: d_f = (1 / s_r) * K 
where s_r is sampling rate, K is frame size. Basically this is (1 / 1 / sampling period) * K = time between two samples * number of samples in a frame

We usually want some overlap to the frames.

Other useful resources for this: 
https://shabda.readthedocs.io/en/latest/references/AudioBasics.html
    


## Librosa's definition:
> A frame is a short slice of a time series used for analysis purposes. This usually corresponds to a single column of a spectrogram matrix.

Or:

> Frames here correspond to short windows of the signal (y, our 1D array time series), each separated by hop_length = 512 samples. librosa uses centered frames, so that the kth frame is centered around sample k * hop_length.




# Fourier transform
Quote from Mueller book:

* "the Fourier transform converts a time- dependent signal into a frequency-dependent function. 
* The inverse process is realized by the Fourier representation, which represents a signal as a weighted superposition of independent elementary functions.
* Each of the weights expresses the extent to which the corresponding elementary function contributes to the original signal, thus revealing a certain aspect of the signal. Because of their explicit physical interpretation in terms of frequency, sinusoids are particularly suited to serve as elementary functions. 
* Each of the weights is then associated to a frequency value and expresses the degree to which the signal contains a periodic oscillation of that fre- quency. 
The Fourier transform can be regarded as a way to compute the frequency- dependent weights.

# Other useful resources

Meinard Muller's book seems quite good, and pitched at about the right level
https://musicinformationretrieval.com/index.html

https://hackernoon.com/intro-to-audio-analysis-recognizing-sounds-using-machine-learning-qy2r3ufl


# What features to use for audio DL
 
Looks like it's becoming more common to use log spectrogram instead of MFCC:
* https://dsp.stackexchange.com/questions/64647/using-mfccs-for-acoustic-machine-failure-prediction/70504#70504
* Dhruv Jain's paper and other papers I've seen also use log spectrogramf


## Audio Classification Papers

### "Nonverbal Sound Detection for Disordered Speech"

https://homes.cs.washington.edu/~djain/img/talks/ICASSP22_Mouthsounds.mp4
https://homes.cs.washington.edu/~djain/img/portfolio/Lea_PersonalizedVoiceTriggers_ICASSP2022.pdf

#### How this is or is not relevant

* Basically addressing the same sort of problem as us
    - their solution is low latency: average latency is 108ms. But variance is high.
* though two of their constraints are slightly different: 
    - they want their sounds to ones that people with speech disabilities can also produce; this is a good goal for us too, but isn't as strict a constraint
    - they want to be able to do inference with their model on a phone, whereas we're running on a local computer (most likely a laptop)
    

#### Data collection

710 adults, 10 samples each for each sound class


#### Data preprocessing

Annotation:

* Each mouth sound recording contains repeated instances of one sound type with silence in between. 
* Make labels for each frame by computing the energy in the audio signal and finding segments with minimum duration of 30 ms and whose relative energy exceeded one standard deviation from the mean. 
    - All frames within a given segment were labeled with the user-annotated sound type and all others were considered "silence."
* Labels for speech clips were generated using a speech activity detector and all aggressor clip frames were labeled with the background class.

Data pipeline:
    
> 300ms window of 16 hz sound recordings -> 64 dimensional log mel-spectrograms generated with a 25 ms window and stride of 10 ms, resulting in a 100 hz sampling rate.




#### Model





### "HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection"

https://github.com/RetroCirce/HTS-Audio-Transformer

#### How this is or is not relevant

* Colin Lea's talk points out that a lot of previous audio classification models are high latency


#### Data

Trained on AudioSet (20h for each of four 12GB V100 GPUs)

##### Preprocessing

#### Model

31M params
