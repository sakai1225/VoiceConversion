# StarGAN-VC2

## Summary

![](https://github.com/SerialLain3170/VoiceConversion/blob/master/StarGANVC/sat_adv_loss.png)

- There are two contributions to StarGAN-VC, source-and-adversarial loss and conditional instance normalization.
- As minor contributions, authors replace generator with generator in CycleGAN-VC2 (2D-1D-2D structure) and use projection in discriminator.

## Usage

### Training Phase

Execute the command line below. 
```bash
$ python train.py --path <NPY_PATH> --adv_type <ADV_TYPE> --res
```

- `NPY_PATH` is a directory which contains training npy files.  
  - npy file includes spectral envelope extracted by WORLD.
  - I assume the path structure of `NPU_PATH` below.

```
IMG_PATH - dir1 - file1
                - file2
                - ...
                
         - dir2 - file1
                - file2
                - ...
                
         - dir3 - file1
                - file2
                - ...
         - ...
 ```
 
-  `ADV_TYPE` are two options, `sat` and `orig`.
    - `orig`: Original adversarial loss
    - `sat`: Source-and-Target adversarial loss proposed in StarGAN-VC2
  
- `--res` option enbales residual connection of input and output proposed in [this paper](https://www.isca-speech.org/archive/Interspeech_2019/pdfs/2067.pdf)
 
 ## Result 
 Results are in [my blog](https://medium.com/@crosssceneofwindff/stargan-vc2%E3%82%92%E7%94%A8%E3%81%84%E3%81%9F%E8%A4%87%E6%95%B0%E8%A9%B1%E8%80%85%E9%96%93%E5%A3%B0%E8%B3%AA%E5%A4%89%E6%8F%9B-24869af1e122) (in Japanese). Please visit this link.
