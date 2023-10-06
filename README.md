# DNN based score-level fusion

## Setup environment
```
bash 00_envsetup.sh
```
## Training
### Seperate training
```
bash 02_train_category.sh DATA/asvspoof2019 10
bash 02_train_fuse.sh DATA/asvspoof2019 10
```

## Evaluation
```
bash 03_eval.sh
```