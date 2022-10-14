# Notes on audio stuff


## Our specific temporal convolutional net approach

### Questions to address

#### How different is the FSD50K baselines audio data pre-processing going to be from the non-verbal sound pre-processing?

**FSD50K pre-proprocessing:**Basically segment audio into 1-sec mel-spectrograms
* Incoming audio is downsampled to 22.050 kHz and transformed to 96-band, log-mel spectrogram as input representation. 
* To deal with the variable-length clips, they use time-frequency (T-F) patches of 1s (equivalent to 101 frames of 30 ms with 10 ms hop). So the input to all models is of shape TxF=101x96. 
* Clips shorter than 1 s are concatenated until such length, while longer clips are sliced in T-F patches with 50% overlap inheriting the clip-level label (a.k.a. false strong la- beling [79]).

See https://github.com/marc1701/fsd_fed/blob/4a9c91c459c6fb61289ae5952d9ba3d232a55f3d/fed_fsd.py#L278 and 
https://github.com/SarthakYadav/fsd50k-pytorch/blob/49b70777f020a7325b290111145da6c69c281ac3/src/data/dataset.py#L73
https://github.com/SarthakYadav/fsd50k-pytorch/blob/49b70777f020a7325b290111145da6c69c281ac3/src/data/audio_parser.py

**Lea & D. Jain pre-processing:**

When training: concat non-verbal sounds with aggressor sounds; predict the probability of each sound per frame

Let T := the total number of frames in the input audio segment.
* Batches of 50% mouth sound clips and 50% aggressors are concatenated, with cumulative duration of T frames, outputting T log probability vectors, with a loss evaluated at 100 hz before the post-processing function.
* Boundaries of each segment are inflated by 50% of the receptive field size (13 frames) to encourage the model to detect the onset and offset of a sound, where many of the constituent frames are “silence”. This is equivalent to the temporal augmentation used by Meyer et al. [21].

"The input 64 dimensional log mel-spectrograms generated from 16k hz audio with a 25 ms window and stride of 10 ms, resulting in a 100 hz sampling rate"


The following papers seem helpful for trying to figure out exactly how to do the data preprocessing if we care about making it low latency:

* A low-latency real-time-capable singing voice detection method with LSTM recurrent neural networks


### Process


1. Train a TCN model on FSD50k data








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


# Data augmentation

One way to combine nnAudio with torch-audiomentations: https://github.com/KinWaiCheuk/nnAudio/issues/49


# nnAudio-related notes

Tutorial: Build a deep neural network for the keyword spotting (KWS) task with nnAudio GPU audio processing
* https://github.com/heungky/nnAudio_tutorial/tree/ef56bf21441b01114f7bd4d165fc1a7a039c8d33

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


Nice walk through of a simple augmentation + pytorch model workflow: https://ketanhdoshi.github.io/Audio-Classification/


TRAINING NEURAL AUDIO CLASSIFIERS WITH FEW DATA
github.com/jordipons/neural-classifiers-with-few-audio

# What features to use for audio DL
 
Looks like it's becoming more common to use log spectrogram instead of MFCC:
* https://dsp.stackexchange.com/questions/64647/using-mfccs-for-acoustic-machine-failure-prediction/70504#70504
* Dhruv Jain's paper and other papers I've seen also use log spectrogram


# Temporal Conv nets

Brief writeup: https://dida.do/blog/temporal-convolutional-networks-for-sequence-modeling

https://arxiv.org/abs/1803.01271


https://www.kaggle.com/code/ceshine/pytorch-temporal-convolutional-networks/script

# Whisper and other large pretrained models


## Whisper
Ryan's note on latency: "Whisper was painfully slow compared to the other models tested. I achieved much higher throughput when running my GPU tests on the largest Talon 1B model and Nemo xlarge (600M) model than any Whisper model, including Whisper Tiny (39M)."


## Nvidia Nemo

https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Online_ASR_Microphone_Demo.ipynb

https://wandb.ai/aarora/Nvidia%20NeMO/reports/Train-Optimize-Analyze-Visualize-and-Deploy-Models-for-Automatic-Speech-Recognition-with-NVIDIA-s-NeMo--VmlldzoxNzI0ODEw

Using whisper for real-time tasks: https://github.com/openai/whisper/discussions/2
    * "It doesn't support real-time per se, but you could build something similar by e.g. incrementally transcribing the audio every second"
    * Streaming transcriber with whisper: https://github.com/shirayu/whispering
    * https://github.com/ggerganov/whisper.cpp (c++, lol)
    * https://github.com/saharmor/whisper-playground



# FSD50K-related scripts

## fsd_fed
https://github.com/marc1701/fsd_fed
The main features of the code provided here are:

* Script performing segmentation of FSD50K dataset into 101x96 mel-spectrograms as specified in the original FSD50K paper.
* Reformatting of metadata to include uploader info and number of created segments per original clip.
* PyTorch Dataset objects for the FSD50K mel-spectrogram data.
* Functions to train models using FL (federated learning).
* Script utilising ray[tune] to perform a grid search of FL parameters.
* Bash dataset downloading script

## unofficial pytorch FSD50K baslines implementation

https://github.com/SarthakYadav/fsd50k-pytorch

## Audio Classification Modelling Papers / Resources

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
    
> 300ms window [?] of 16 hz sound recordings -> 64 dimensional log mel-spectrograms generated with a 25 ms window and stride of 10 ms, resulting in a 100 hz sampling rate.

> The first layers apply 1D convolu- tions (kernel size k=5) with N=256 nodes.




#### Model

Input to model: 300ms audio
Model: Temporal Convolutional Network with Dilated ResNet architecture



### "HTS-AT: A Hierarchical Token-Semantic Audio Transformer for Sound Classification and Detection"

https://github.com/RetroCirce/HTS-Audio-Transformer

#### How this is or is not relevant

* Colin Lea's talk points out that a lot of previous audio classification models are high latency


#### Data

Trained on AudioSet (20h for each of four 12GB V100 GPUs)

##### Preprocessing

#### Model

31M params


### Kaggle Freesound tagging

SPECMIX: A SIMPLE DATA AUGMENTATION AND WARM-UP PIPELINE
TO LEVERAGE CLEAN AND NOISY SET FOR EFFICIENT AUDIO TAGGING
https://github.com/ebouteillon/freesound-audio-tagging-2019

used a vgg-like model

### Kaggle Bird Sound classification

#### Conde, Shubham, et al

http://ceur-ws.org/Vol-2936/paper-131.pdf
https://github.com/kumar-shubham-ml/kaggle-birdclef-2021 

They used data agumentation on the log mel-spectrograms (as opposed to the waveforms), with a probability between 0.4 and 0.7 

#### Murakami et al

http://ceur-ws.org/Vol-2936/paper-136.pdf




https://github.com/daisukelab/ml-sound-classifier


