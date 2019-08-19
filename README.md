# A Deep Motion Deblurring Network based on Per-Pixel Adaptive Kernels with Residual Down-Up and Up-Down Modules
A source code of the 3rd winner of NTIRE 2019 Video Deblurring Challenge (*CVPRW*, 2019) : 
"A Deep Motion Deblurring Network based on Per-Pixel Adaptive Kernels with Residual Down-Up and Up-Down Modules" by Hyeonjun Sim and Munchurl Kim. [[pdf](http://openaccess.thecvf.com/content_CVPRW_2019/papers/NTIRE/Sim_A_Deep_Motion_Deblurring_Network_Based_on_Per-Pixel_Adaptive_Kernels_CVPRW_2019_paper.pdf)], [[NTIRE2019](http://www.vision.ee.ethz.ch/ntire19/)]

## Prerequisites
* python 2.7
* tensorflow (gpu version) >= 1.6 (The runtime in the paper was based on tf 1.6. But the code in this repo also runs in tf 1.13 )

## Testing with pretrained model
We provide the test model with checkpoint in `/checkpoints/` in this repo.
To run the code, 
```bash
python main.py --test_data_path './Dataset/test/' --working_directory './data/'
```
with geometric self-ensemble (takes much more time), 
```bash
python main.py --test_data_path './Dataset/test/' --working_directory './data/' --ensemble
```

`test_data_path` is the location of the test input blur frames
```
├──── Dataset/
   ├──── Video0/
      ├──── 0000.png
      ├──── 0001.png
      └──── ...
   ├──── Video1/
      ├──── 0000.png
      ├──── 0001.png
      └──── ...
   ├──── ...
```
`working_directory` is the location of the deblurred output frames.
```
├──── data/
   ├──── test/
     ├──── Video0/
        ├──── 0000.png
        ├──── 0001.png
        └──── ...
     ├──── Video1/
        ├──── 0000.png
        ├──── 0001.png
        └──── ...
     ├──── ...
```

## Reference
```bibtex
@inproceedings{sim2019deep,
  title={A Deep Motion Deblurring Network Based on Per-Pixel Adaptive Kernels With Residual Down-Up and Up-Down Modules},
  author={Sim, Hyeonjun and Kim, Munchurl},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2019}
}
```
## Contact
Please send me an email, flhy5836@kaist.ac.kr
