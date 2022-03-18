# YOLOF: You Only Look At One-level Feature

This is a lite version of YOLOF, which is built by PyTorch.

# What does 'Lite' mean?
Different from the official YOLOF where the input images are resized to have their shorter side being 800 and their longer side less or equal to 1333, 
the aspect ratio is not kept in my YOLOF-Lite. In official YOLOF, the aspect ratio is kept. For example, given an input image, I just resize to a shape like 800×800 or 640×640.
Compared to the official method, my resize method might lead to a bit lower performance, but achieving the same performance is not my goal. 

I think it might be easy to deploy the resize method used in YOLOF in my YOLOF-Lite. I highly encourage you to try it out.

For other details, I try to align my configuration with official YOLOF.

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

# Main results on COCO-val

| Model                                     |  Size       |   mAP   |  FPS  |  GFLOPs |  #params | Weight |
|-------------------------------------------|-------------|---------|-------|---------|----------|--------|
| YOLOF_R_18_C5_1x                          |  928 × 928  |  31.7   |   80  |  47     |  31M     | [github](https://github.com/yjh0410/YOLOF-Lite/releases/download/YOLOF-Lite-Weight/yolof_r18_C5_1x_31.7.pth) |
| YOLOF_R_50_C5_1x                          |  928 × 928  |  37.4   |   34  |  86     |  44M     | [github](https://github.com/yjh0410/YOLOF-Lite/releases/download/YOLOF-Lite-Weight/yolof_r50_C5_1x_37.4.pth) |
| YOLOF_R_50_DC5_1x                         |  928 × 928  |         |       |  171    |  44M     |        |
| YOLOF_R_101_C5_1x                         |  928 × 928  |         |       |  150    |  63M     |        |
| YOLOF_CSP_D_53_DC5_3x                     |  608 × 608  |         |       |  194    |  45M     |        |
| YOLOF_CSP_D_53_DC5_9x                     |  608 × 608  |         |       |  194    |  45M     |        |

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
