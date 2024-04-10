# This is the official implementation of BDRNet
🔵## Training Preparation
### Environment
☑️pytorch 1.8.1  
☑️CUDA 11.2  
☑️python 3.8.10  
### Requirements
run ```pip install -r requirements.txt```
### DataSet Example
Click [here](https://lafi.github.io/LPN/) to download the PUCPR dataset.  
Then, modify the default path of the dataset in [setting.py](datasets/PUCPR/setting.py).
### Available GPU IDs
Modify the available GPU devices in [config.py](config.py)
🔵## Training
run ```python train.py```  
The training log is saved in ```./exp/Your_EXP_Name/code/log/training.txt```
🔵## Overall Results:
### Figure
![overall results](results1.jpg)
### Table
|Dataset|MAE|MSE|
| :---:         |     :---:      |          :---: |
|PUCPR|1.55|1.93|
