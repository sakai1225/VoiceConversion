# CycleGAN-VC2

## Summary

![](https://github.com/SerialLain3170/VoiceConversion/blob/master/CycleGANVC2/network.png)

- There are three contributions to original CycleGAN-VC. Two-step adversarial loss, 2D-1D-2D generator and PatchGAN.

## Usage

### Training Phase

Execute the command line below. 
```bash
$ python train.py --src <SRC_NPY_PATH> --tgt <TGT_NPY_PATH> --second
```

- `SRC_NPY_PATH` is a directory which contains source npy files.  
- `TGT_NPY_PATH` is a directory which contains target npy files.
  - npy file includes spectral envelope extracted by WORLD.

- `--second` option enables two-adversarial loss. However, as far as I experimented, result samples have better quality if I apply one-step adversarial loss (like original CycleGAN).

## Result
Result samples are in [my blog](https://medium.com/@crosssceneofwindff/%E7%BE%8E%E5%B0%91%E5%A5%B3%E5%A3%B0%E3%81%B8%E3%81%AE%E5%A4%89%E6%8F%9B%E3%81%A8%E5%90%88%E6%88%90-fe251a8e6933) (in Japanese). Please visit this link.
