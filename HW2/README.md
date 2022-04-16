# CIFAR100_Classification


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

### For EfficientNet_b4 with MCCE loss
```
python --model EfficientB4
```

### For Swin Transformer
```
python --model Swin
```


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
    <td>RandomHorizontalFlip,<br>RandomRotation,<br>Normalize</td>
    <td>84.27 </td>
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
