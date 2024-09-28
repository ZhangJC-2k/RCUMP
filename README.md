This is a repo for our work: "**[RCUMP: Residual Completion Unrolling with Mixed Priors for Snapshot Compressive Imaging](https://ieeexplore.ieee.org/abstract/document/10471309)**"

### News
Our work has been accepted by TIP, codes and results are coming soon (May or June).

### Codes and Results
The codes have been released and more details will be updated in a few days.

The simulated and real results of RCUMP are available [here](https://pan.baidu.com/s/18FBrgicFiXT1wOYr2ZZ7OQ?pwd=rcum).

### 1. Environment Requirements
```shell
Python>=3.6
scipy
numpy
```

### 2. Train:

Download the cave dataset of [MST series](https://github.com/caiyuanhao1998/MST) from [Baidu disk](https://pan.baidu.com/s/1X_uXxgyO-mslnCTn4ioyNQ?pwd=fo0q)`code:fo0q` or [here](https://pan.baidu.com/s/1gyIOfmUWKrjntKobUjwTjw?pwd=lup6), put the dataset into the corresponding folder "RCUMP/CAVE_1024_28/" as the following form:

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

Download the test dataset from [here](https://pan.baidu.com/s/1KqMo3CY8LU9HRU2Lak9yfQ?pwd=c0a2), put the dataset into the corresponding folder "RCUMP/Test_data/" as the following form:

	|--Test_data
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

### Citation
If this repo helps you, please consider citing our work:


```shell
@ARTICLE{RCUMP,
  author={Zhao, Yin-Ping and Zhang, Jiancheng and Chen, Yongyong and Wang, Zhen and Li, Xuelong},
  journal={IEEE Transactions on Image Processing}, 
  title={RCUMP: Residual Completion Unrolling With Mixed Priors for Snapshot Compressive Imaging}, 
  year={2024},
  volume={33},
  number={},
  pages={2347-2360},
  keywords={Imaging;Image coding;Iterative methods;Optimization;Image reconstruction;Hyperspectral imaging;Artificial neural networks;Snapshot compressive imaging;hyperspectral image;deep unrolling-based methods},
  doi={10.1109/TIP.2024.3374093}}
```
