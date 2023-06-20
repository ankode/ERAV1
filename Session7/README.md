# ERAV1 SESSION 7 ASSIGNMENT

This folder contains 3 python notebook. Each notebook gets model from model.py.

*model.py* contains 3 model whose details are mentiond below:

## Model 1

**Target:** 
```
1.Achieve accuray with 12k parameters 
2. Added Dropout
```

**Results:**
```
- Parameters: 12,672
- Best Train Accuracy: 99.34
- Best Test Accuracy: 99.36 (11th Epoch)
```
**Analysis:**
```
1. Dropout overcomes overfitting
```


## Model 2

**Target:** 
```
1. Add Data Augmentation
2. Make model lighter

```

**Results:**
```
- Parameters: 8,896
- Best Train Accuracy: 98.70
- Best Test Accuracy: 99.20 (9th Epoch)
```
**Analysis:**
```
1. Decreasing the number of channels makes model lightweight
2. Adding Data Augmentation helps overcome over fitting.
```

## Model 3

**Target:**
```
1. Add  StepLR
2. Make model lighter
```

**Results:**
```
- Parameters: 7,680
- Best Train Accuracy: 98.98
- Best Test Accuracy: 99.52 (7th Epoch)
```
**Analysis:**
```
1. Consistent 99.4%+ accuracy with max accuracy 99.52%
2. Decreasing the number of channels and increasing the layers seems to be good way to make model lighter
3. Having stepLR with step size 5 to adjust learning rate helps in learning.
```