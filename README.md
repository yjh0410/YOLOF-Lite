# YOLOQ: Single level Object Detector with Dynamic Attention
# Requirements
- We recommend you to use Anaconda to create a conda environment:
```Shell
conda create -n yoloq python=3.6
```

- Then, activate the environment:
```Shell
conda activate yoloq
```

- Requirements:
```Shell
pip install -r requirements.txt 
```
PyTorch >= 1.9.1 and Torchvision >= 0.10.3

# Visualize positive sample
You can run following command to visualize positiva sample:
```Shell
python train.py \
        -d voc \
        --batch_size 2 \
        --root path/to/your/dataset \
        --vis_targets
```

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
