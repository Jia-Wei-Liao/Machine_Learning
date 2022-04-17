# CIFAR100_Classification

## Requirements
- `numpy == `
- `torch == `
- `torchvision ==`
- `PIL`
- `tqdm`
- `scipy`
- `timm`


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
<table>
  <tr>
    <td>Model</td>
    <td>Batch size</td>
    <td>Epochs</td>
    <td>Loss</td>
    <td>Optimizer</td>
    <td>Scheduler</td>
    <td>Augmentation</td>
    <td>test acc</td>
  </tr>
  <tr>
    <td>EfficientNet-b4</td>
    <td>32</td>
    <td>200</td>
    <td>CE</td>
    <td>AdamW (lr=1e-3,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>RandomHorizontalFlip,<br>RandomRotation,<br>Autoaugmentation,<br>RandomNoise,<br>Normalize</td>
    <td>88.57% </td>
  </tr>
  <tr>
    <td>EfficientNet-b4</td>
    <td>64</td>
    <td>200</td>
    <td>CE</td>
    <td>AdamW (lr=1e-3,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>RandomHorizontalFlip,<br>RandomRotation,<br>Autoaugmentation,<br>RandomNoise,<br>Normalize</td>
    <td>89.34% </td>
  </tr>
  <tr>
    <td>EfficientNet-b4</td>
    <td>64</td>
    <td>200</td>
    <td>MCCE</td>
    <td>AdamW (lr=1e-3,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>RandomHorizontalFlip,<br>RandomRotation,<br>Autoaugmentation,<br>RandomNoise,<br>Normalize</td>
    <td>88.73% </td>
  </tr>
  <tr>
    <td>EfficientNet-b4</td>
    <td>64</td>
    <td>200</td>
    <td>FL</td>
    <td>AdamW (lr=1e-3,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>RandomHorizontalFlip,<br>RandomRotation,<br>Autoaugmentation,<br>RandomNoise,<br>Normalize</td>
    <td>89.35% </td>
  </tr>
  <tr>
    <td>EfficientNet-b4</td>
    <td>32</td>
    <td>200</td>
    <td>FLSD53</td>
    <td>AdamW (lr=1e-3,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>RandomHorizontalFlip,<br>RandomRotation,<br>Autoaugmentation,<br>RandomNoise,<br>Normalize</td>
    <td>88.90% </td>
  </tr>
  <tr>
    <td>EfficientNet-b4</td>
    <td>32</td>
    <td>200</td>
    <td>FLSD53</td>
    <td>AdamW (lr=1e-3,  weight decay=1e-4)</td>
    <td>Step (size=3, gamma=0.8)</td>
    <td>RandomHorizontalFlip,<br>RandomRotation,<br>Autoaugmentation,<br>RandomNoise,<br>Normalize</td>
    <td>89.52% </td>
  </tr>
</table>


## GitHub Acknowledgement
We thank the authors of these repositories:
- MCCE: https://github.com/Kurumi233/Mutual-Channel-Loss  
- FLSD: https://github.com/torrvision/focal_calibration  
- Ranger21: https://github.com/lessw2020/Ranger21  
- AutoAugment: https://github.com/DeepVoltaire/AutoAugment  


## Citation
```
@misc{
    title  = {cifa100_classification},
    author = {Jia-Wei Liao},
    url    = {https://github.com/Jia-Wei-Liao/Machine_Learning/tree/main/HW2},
    year   = {2022}
}
```
