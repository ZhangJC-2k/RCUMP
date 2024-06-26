This is a repo for our work: "**[RCUMP: Residual Completion Unrolling with Mixed Priors for Snapshot Compressive Imaging](https://ieeexplore.ieee.org/abstract/document/10471309)**"

### News
Our work has been accepted by TIP, codes and results are coming soon (May or June).

### Codes and Results
The codes have been released and more details will be updated in a few days.

The simulated and real results of RCUMP are available in [Baidu Disk](https://pan.baidu.com/s/18FBrgicFiXT1wOYr2ZZ7OQ?pwd=rcum).

### 1. Environment Requirements
```shell
Python>=3.6
scipy
numpy
```

### 2. Train:

Download the cave dataset of size 1024 from [MST](https://github.com/caiyuanhao1998/MST), put the dataset into the corresponding folder "RCUMP/CAVE_1024_28/" as the following form:

	|--CAVE_1024_28
        |--scene1.mat
        |--scene2.mat
        ：
        |--scene205.mat
        |--train_list.txt
Then run the following command
```shell
cd RCUMP
python Train.py
```

### 3. Test:

Download the dataset from [TSA-Net](https://github.com/mengziyi64/TSA-Net), put the dataset into the corresponding folder "RCUMP/Test_data/" as the following form:

	|--Datasets
        |--scene01.mat
        |--scene02.mat
        ：
        |--scene10.mat
        |--test_list.txt
Then run the following command
```shell
cd RCUMP
python Test.py
```
Finally, run 'cal_psnr_ssim.m' in Matlab to get the performance metrics.
