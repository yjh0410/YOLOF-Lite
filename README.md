# YOLOF: You Only Look At One-level Feature

This is a lite version of YOLOF, which is built by PyTorch.

# What does 'Lite' mean?
Different from the official YOLOF where the input images are resized to have their shorter side being 800 and their longer side less or equal to 1333, 
the aspect ratio is not kept in my YOLOF-Lite. In official YOLOF, the aspect ratio is kept. For example, given an input image, I just resize to a shape like 800×800 or 640×640.

I think it might be easy to deploy the resize method used in YOLOF in my YOLOF-Lite. If you
want to try, just try.

# Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n yolof python=3.6
```

- Then, activate the environment:
```Shell
conda activate yolof
```

- Requirements:
```Shell
pip install -r requirements.txt 
```
PyTorch >= 1.9.1 and Torchvision >= 0.10.3

## Main results

| Model                                     |  COCO val mAP |  FPS  |
|-------------------------------------------|---------------|-------|
| YOLOF_R_50_C5_1x                          |    35.6       |   -   |
| YOLOF_R_50_DC5_1x                         |               |       |
| YOLOF_R_101_C5_1x                         |               |       |
| YOLOF_R_101_DC5_1x                        |               |       |
| YOLOF_CSP_D_53_DC5_3x                     |               |       |
| YOLOF_CSP_D_53_DC5_9x                     |               |       |
| YOLOF_CSP_D_53_DC5_9x_stage2_3x           |               |       |

More results are coming ...

# Train
```Shell
sh train.sh
```

You can change the configurations of `train.sh`.

According to your own situation, you can make necessary adjustments to the above run commands

## Test
```Shell
python test.py -d coco \
               --cuda \
               --weight path/to/weight \
               --img_size 640 \
               --root path/to/dataset/ \
               --show
```

You can run the above command to visualize the detection results on the dataset.
