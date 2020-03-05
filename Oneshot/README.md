# One-shot Voice Conversion

## Summary

![](https://github.com/SerialLain3170/VoiceConversion/blob/master/Oneshot/concept.png)

- This is implementation of one-shot voice conversion (using only one target speaker's utterance).
- Authors give utterance data VAE by way of Adaptive Instance Normalization (AdaIN).

## Usage

### Preprocess

1. Download [JVS Corpus](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus)
2. Execute the command line below. This command converts wav files in JVS corpus into mel-spectrogram. 

```bash
python dataset.py --jvs_path <JVS_PATH> --mel_path <MEL_PATH>
```

- `JVS_PATH`: path which indicates JVS corpus (previous downloaded)
- `MEL_PATH`: output directory

### Training Phase
Execute the command line below.

```bash
python train.py --path <MEL_PATH>
```

- `MEL_PATH`: output directory of preprocess

## Result
This implementation doesn't work well.  
For example, when converting male source speaker into female target speaker, the result sounds like woman. However, it doesn't sound like female target speaker's voice itself.
