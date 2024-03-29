[![LinkedIn][linkedin-shield]][linkedin-url-Bucsa]



# Machine-Learning-Fitness-Tracker

## Goal

Create Python scripts to process, visualize, and model accelerometer and gyroscope data to create a machine learning model that can classify barbell exercises and count repetitions.

## Terminal Commands

### Creating the VENV environment

```bash
py -m venv venv
```

    
### Activating the VENV environment

- For Command Prompt - cmd
    
```bash

C:\Users\User\Machine-Learning-Fitness-Tracker> venv\Scripts\activate.bat

(venv) C:\Users\User\Machine-Learning-Fitness-Tracker> 

```
   
- For Git Bash - bash
    
```bash
User@User MINGW64 ~/Machine-Learning-Fitness-Tracker (main)
$ source venv/Scripts/activate

(venv) 
User@User MINGW64 ~/Machine-Learning-Fitness-Tracker (main)
$ 

``` 

### Deactivating the VENV environment


- For Command Prompt - cmd

```bash
(venv) C:\Users\User\Machine-Learning-Fitness-Tracker> venv\Scripts\deactivate.bat

C:\Users\User\Machine-Learning-Fitness-Tracker> 

```
   
- For Git Bash - bash
    
```bash
(venv) 
User@User MINGW64 ~/Machine-Learning-Fitness-Tracker (main)
$ deactivate

User@User MINGW64 ~/Machine-Learning-Fitness-Tracker (main)
$ 

``` 


## PIP Installs

What was actually installed
     
```bash
pip install numpy
pip install pandas
pip install ipykernel==6.17.1 
pip install ipython==8.7.0 
pip install jupyter-client==7.4.7 
pip install jupyter-core==5.1.0 
pip install matplotlib
pip install math
pip install spicy
pip install scikit-learn-intelex
```
  
What was should be installed
    
```bash
pip install numpy==1.23.5 
pip install pandas==1.5.2 
pip install ipykernel==6.17.1 
pip install ipython==8.7.0 
pip install jupyter-client==7.4.7 
pip install jupyter-core==5.1.0 
pip install matplotlib==3.6.2
```
Installing Current Versions of ALL PIP files.
```bash
pip install numpy
pip install pandas
pip install ipykernel 
pip install ipython
pip install jupyter-client
pip install jupyter-core
pip install matplotlib
```

## Data Collection

Data was collection from 5 participants: A, B, C, D, and E. 

Data was collection from 01/11/2019 (2019/01/11) to 01/16/2019 (2019/01/16)

Equipment used was MbientLab's wristband sensor research kit
- The wristband mimics the placement and orientation
of a watch while allowing for controlled experiments. Data was collected using the
default settings of the sensors: accelerometer: 12.500HZ and gyroscope: 25.000Hz.

## Understanding the data

Acc_Y 
- This is accerlation in the verticle direction, Up and Down.

Acc_x 
- This is accerlation in the horizontal direction, Left to Right (West to East).

Acc_z 
- This is accerlation in the horizontal direction, Front to Back (North to South).

[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url-Bucsa]: https://www.linkedin.com/in/justin-bucsa