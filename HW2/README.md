# CIFAR100_Classification

## File structure
```
.
├─src
|   ├─losses
|   |   ├─FL.py
|   |   ├─MCCE.py
|   |   └─TSCE.py
|   ├─models
|   |   ├─pretrain_weight
|   |   |    └─pretrain_weight
|   |   |        └─swin_base_patch4_window12_384_22k.pth
|   |   └─swin.py
|   ├─optimizer
|   |   ├─chebyshev_lr_functions.py
|   |   ├─ranger21.py
|   |   └─rangerabel.py
|   ├─dataset.py
|   ├─logger.py
|   ├─trainer.py
|   ├─transforms.py
|   └─utils.py
├─config.py
├─model.py
├─test.py
└─train.py
```

## Download Swin Transformer pre-trained weight
- You can download the pre-trained weight on the Google Drive:
https://drive.google.com/file/d/1iMU08acRJNOIpX0La0kLAwFbEwq3dosO/view?usp=sharing
- You can download the pre-trained weight on the Swin Transformer GitHub:
https://github.com/microsoft/Swin-Transformer

## Requirements
- `numpy == 1.21.5`
- `torch == 1.10.2`
- `torchvision == 0.11.3`
- `PIL == 9.0.1`
- `tqdm == 4.63.0`
- `scipy == 1.7.3`
- `timm == 0.4.12`


## Training
To train the model, you can run this command:
```
python train.py -bs <batch size> \
                -ep <epoch> \
                --model <model> \
                --loss <loss function> \
                --optim <optimizer> \
                --lr <learning rate> \
                --weight_decay <parameter weight decay> \
                --scheduler <learning rate schedule> \
                --autoaugment <use autoaugmentation> \
                --rot_degree <degree of rotation> \
                --fliplr <probabiliy of horizontal flip> \
                --noise <probabiliy of adding gaussian noise> \
                --num_workers <number worker> \
                --device <gpu id> \
                --seed <random seed>
```
- model: EfficientB4, Swin
- loss: CE, MCCE, FL, FLSD
- optim: SGD, Adam, AdamW, Ranger
- scheduler: step (gamma, step_size), cos


## Testing
To test the results, you can run this command:
```
python test.py
```

## Experiment results
Before training, we use random Horizontal flip, random rotation, auto-augmentation, random noise, normalize as preprocessing. 
<table>
  <tr>
    <td>Model</td>
    <td>Batch size</td>
    <td>Epochs</td>
    <td>Loss</td>
    <td>Optimizer</td>
    <td>Scheduler</td>
    <td>test acc</td>
  </tr>
  <tr>
    <td>EfficientNet-B4</td>
    <td>32</td>
    <td>200</td>
    <td>CE</td>
    <td>AdamW (lr=1e-3,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>88.57% </td>
  </tr>
  <tr>
    <td>EfficientNet-B4</td>
    <td>64</td>
    <td>200</td>
    <td>CE</td>
    <td>AdamW (lr=1e-3,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>89.34% </td>
  </tr>
  <tr>
    <td>EfficientNet-B4</td>
    <td>32</td>
    <td>200</td>
    <td>MC</td>
    <td>AdamW (lr=1e-3,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>88.37% </td>
  </tr>
  <tr>
    <td>EfficientNet-B4</td>
    <td>64</td>
    <td>200</td>
    <td>MC</td>
    <td>AdamW (lr=1e-3,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>88.73% </td>
  </tr>
  <tr>
    <td>EfficientNet-B4</td>
    <td>32</td>
    <td>200</td>
    <td>FL</td>
    <td>AdamW (lr=1e-3,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>88.24% </td>
  </tr>
  <tr>
    <td>EfficientNet-B4</td>
    <td>64</td>
    <td>200</td>
    <td>FL</td>
    <td>AdamW (lr=1e-3,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>89.35% </td>
  </tr>
  <tr>
    <td>EfficientNet-B4</td>
    <td>32</td>
    <td>200</td>
    <td>FLSD53</td>
    <td>AdamW (lr=1e-3,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>88.90% </td>
  </tr>
  <tr>
    <td>EfficientNet-B4</td>
    <td>64</td>
    <td>200</td>
    <td>FLSD53</td>
    <td>AdamW (lr=1e-3,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>89.52% </td>
  </tr>
  <tr>
    <td>Swin-B</td>
    <td>64</td>
    <td>200</td>
    <td>FL</td>
    <td>AdamW (lr=3e-5,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>93.56% </td>
  </tr>
  <tr>
    <td>Swin-B</td>
    <td>64</td>
    <td>200</td>
    <td>FLSD53</td>
    <td>AdamW (lr=3e-5,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>93.60% </td>
  </tr>
  <tr>
    <td>Teacher-Student</td>
    <td>64</td>
    <td>200</td>
    <td>FL</td>
    <td>AdamW (lr=3e-5,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>90.10% </td>
  </tr>
</table>


## GitHub Acknowledgement
We thank the authors of these repositories:
- AutoAugmentation: https://github.com/DeepVoltaire/AutoAugment  
- Swin Transformer: https://github.com/microsoft/Swin-Transformer  
- MC-Loss: https://github.com/Kurumi233/Mutual-Channel-Loss  
- FLSD53: https://github.com/torrvision/focal_calibration  
- Ranger21: https://github.com/lessw2020/Ranger21  


## Citation
```
@misc{
    title  = {cifa100_classification},
    author = {Jia-Wei Liao},
    url    = {https://github.com/Jia-Wei-Liao/Machine_Learning/tree/main/HW2},
    year   = {2022}
}
```
