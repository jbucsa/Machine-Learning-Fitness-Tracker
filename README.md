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
pip install seaborn
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


## Steps to Processing Sensor Data and Utilizing Machine Learning Modeling
 
### Step 1: Collecting RAW Sensor Data
1. Data for this project was collected in a ```.csv``` format.
2. Understanding the CSV files (measurement, participant, exercise, intensity)
3. RAW data can be found in the following folder for this project
   ```/data/raw/MetaMotion```

### Step 2: Create (Make) Dataset
1. Create a python file that will import the RAW data from the .csv files and turn it into an data set that can be exportable and thus used through the project. 
2. File for this step can be found
   ```/src/data/make_dataset.py```
Steps within the ```make_dataset.py``` goes as:
   1. Read single CSV file
   2. List all data in data/raw/MetaMotion 
   3. Extract features from filename
   4. Read all files
   5. Working with datetimes
   6. Turn into function
   7. Merging datasets
   8. Resample data (frequency conversion)
   9. Export dataset
3. Export Dataset, for this project we are using pickle (```.pkl```) files to hold dataset files. The pickle (```.pkl```) file for this step can be found
   ```/data/interim/01_data_processed.pkl```
4. Do not forget to add a ```__init__.py``` file for the  ```make_dataset.py```. The ```__init__.py``` for this project can be found 
   ```/src/data/__init__.py```

### Step 3: Visualizing the Sensor Data
1. Create plot settings for that match what is needed to plot the data from created pickle (```.pkl```) from Step 1, found ```./data/interim/01_data_processed.pkl```. The ```plot_settings.py``` can be found
    ```/src/visualization/plot_settings.py```
2. Write a python script that will create figures based on the perviously created pickle (```.pkl```) from Step 1, found ```/data/interim/01_data_processed.pkl```. The ```visualize.py``` can be found
     ```/src/visualization/visualize.py``` 
Steps within the ```visualize.py``` goes as:
   1. Load data
   2. Plot single columns
   3. Plot all exercises
   4. Adjust plot settings
   5. Compare medium vs. heavy sets
   6. Compare participants
   7. Plot multiple axis
   8. Create a loop to plot all combinations per sensor
   9. Combine plots in one figure
   10. Loop over all combinations and export for both sensors
3.  Loop over all combinations and export for both sensors data types. The created graphs can be found in the following folder as ```.png``` files
   ```/reports/figures/```
4.  Do not forget to add a ```__init__.py``` file, it will help the  ```visualize.py``` and ```plot_settings.py``` files. The ```__init__.py``` for this project can be found 
   ```/src/visualization/__init__.py```

### Step 4: Detecting Outliers in Sensor Data
1. Write a python script that will remove the Outliers from the existing Sensor Data found in a pickle (```.pkl```) from Step 1, found ```/data/interim/01_data_processed.pkl```. Label this python file ```remove_outliers.py```, which can be found
    ```/src/features/remove_outliers.py```
Steps within the ```remove_outliers.py``` goes as:
   1. Load data
   2. Plotting outliers
      1. ```plot_binary_outliers()```
   3. Interquartile Range (Distribution Based)
      1. Insert IQR function
      2. ```mark_outliers_iqr()```
   4. Chauvenets Criteron (Distribution Based)
      1. ```mark_outliers_chauvenet()```
   5. Local outlier factor (distance based)
      1. Insert Local Outlier Factor (LOF) function
      2. ```mark_outliers_lof()```
   6. Check outliers grouped by label
   7. Choose method and deal with outliers
   8. Export new dataframe
2. Export the new Dataset, that has outliers from using pickle (```.pkl```) files to hold dataset files. The pickle (```.pkl```) file for this step can be found
   ```/data/interim/02_outliers_removed_chauvenets.pkl```
3. Do not forget to add a ```__init__.py``` file, it will help the  ```remove_outliers.py``` files. The ```__init__.py``` for this project can be found 
   ```/src/features/__init__.py```

### Step 5: Features to Organize and Identify Senor Data.
1.  Import from GitHub code or copy following python files, as there hold functions we will utilized in the following steps.
   ```/src/features/DataTransformation.py```
   ```/src/features/FrequencyAbstraction.py```
   ```/src/features/TemporalAbstraction.py```
2. Now we will write a python file that will first filter subtle noise (not outliers) and identify parts of the data that explain most of the variance. Then add numerical, temporal, frequency, and cluster features. Label this file ```build_features.py```. For this project, the ```build_features.py``` can be found 
    ```/src/features/build_features.py```
Steps within the ```build_features.py``` goes as:
   1. Load data
   2. Dealing with missing values (imputation)
      1. INTERPOLATE will fill in the gap were the data is missing.
   3. Calculating set duration
   4. Butterworth lowpass filter - Low Pass Filter
      1. ```LowPassFilter()```
   5. Principal component analysis PCA
      1. ```PrincipalComponentAnalysis()```
   6. Sum of squares attributes
   7. Temporal abstraction
      1. ```NumericalAbstraction()```
   8. Frequency features - Fourier Transformation
      1. ```FourierTransformation()```
   9. Visualizing Results
   10. Dealing with overlapping windows
   11. Clustering - K-means Clustering
       1.  ```KMeans()```
   12. Export dataset
3. Export the new dataset noting the clusters created as a using pickle (```.pkl```) files to hold dataset files. The pickle (```.pkl```) file for this step can be found
   ```/data/interim/03_data_features.pkl```  
4. No need to create a ```__init__.py``` file for this folder because we did so in Step 4. 




[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url-Bucsa]: https://www.linkedin.com/in/justin-bucsa