# Voice Conversion

## Introduction
I have a desire to convert multi anime characters' voices into another multi anime characters' voices. Of course, I also have a desire to convert my male voice into another cute female voice. when using deep learning, there are mainly two types of voice conversion methods, parallel and non-parallel. In parallel voice conversion, pair of source speaker's voice and target speaker's voice is needed. In the context, pair means both source voice and target voice are made by reading the same sentence and don't match in time domain. However, limitation speakers have to read the same sentences prevents me from collecting data because few paired anime characters' voices exist. Therefore, I try to implement non-parallel voice conversion. 

## Experiment

### Motivation
I try to implement non-parallel voice conversion described above. However, I think that result by non-parallel method is inferior to it by parallel method. Therefore, I experiment various non-parllel voice conversion methods to confirm whether non-parallel method works well comparable to paralle method or not. 

### Methods

- [x] CycleGAN-VC2
- [ ] CycleGAN using mel-spectrogram
- [ ] CycleGAN using linear spectrogram
- [x] StarGAN-VC2
- [ ] StarGAN using mel-spectrogram
- [ ] StarGAN using linear spectrogram
- [ ] Blow
- [ ] WaveCycleGAN
- [x] One shot voice conversion
