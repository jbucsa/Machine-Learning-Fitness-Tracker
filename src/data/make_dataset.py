import pandas as pd
from glob import glob

""" 
--------------------------------------------------------------
    Read single CSV file
--------------------------------------------------------------
"""

single_file_accelerometer = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
    )

single_file_gyroscope = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
    )

"""
--------------------------------------------------------------
    List all data in data/raw/MetaMotion
--------------------------------------------------------------
"""

# List ALL files
filesAll = glob("../../data/raw/MetaMotion/*")

# List ONLY file with CVS as an extension
filesCSV = glob("../../data/raw/MetaMotion/*.csv")
len(filesCSV)

"""
--------------------------------------------------------------
    Extract features from filename
--------------------------------------------------------------
"""

data_path_CSV = "../../data/raw/MetaMotion/*.csv"
data_path = "../../data/raw/MetaMotion\\"
f_CVS = filesCSV[0]

participant = f_CVS.split("-")[0].replace(data_path, "")
label = f_CVS.split("-")[1]
category = f_CVS.split("-")[2].rstrip("1234").rstrip("_MetaWear_2019")

df = pd.read_csv(f_CVS)

# Adding Row to the df [DataFrame]

df["participant"] = participant
df["label"] = label
df["category"] = category


"""
--------------------------------------------------------------
    Read all files
--------------------------------------------------------------
"""

accelerometer_df = pd.DataFrame()
gyroscope_df = pd.DataFrame()

accelerometer_set = 1
gyroscope_set = 1

for x in filesCSV:    
    data_path = "../../data/raw/MetaMotion\\"
    
    participant = x.split("-")[0].replace(data_path, "")
    label = x.split("-")[1]
    category = x.split("-")[2].rstrip("1234").rstrip("_MetaWear_2019")
    
    df = pd.read_csv(x)
    
    df["participant"] = participant
    df["label"] = label
    df["category"] = category

    if "Accelerometer" in x:
        df["set"] = accelerometer_set
        accelerometer_set += 1
        accelerometer_df = pd.concat([accelerometer_df, df])
    
    if "Gyroscope" in x:
        df["set"] = gyroscope_set
        gyroscope_set += 1
        gyroscope_df = pd.concat([gyroscope_df, df])

# Simple code to locate the set value 
accelerometer_df[accelerometer_df["set"] == 35]

"""  
    for f_CVS in filesCSV:
        participant = f_CVS.split("-")[0].replace(data_path, "")
        label = f_CVS.split("-")[1]
        category = f_CVS.split("-")[2].rstrip("1234").rstrip("_MetaWear_2019")
"""


"""
--------------------------------------------------------------
    Working with datetimes
--------------------------------------------------------------
"""

accelerometer_df.info()

# Conversion for UNIX time
pd.to_datetime(df["epoch (ms)"], unit="ms")

""" 
    Conversion for normal time. Note, units not needed. 
    pd.to_datetime(df["time (01:00)"]).dt.month
    Note here, we are reassigning the index (KEY) values from a basic interger to a time value. Note, we have a Column (VALUE) that shares the same label name.
""" 
accelerometer_df.index = pd.to_datetime(accelerometer_df["epoch (ms)"], unit="ms")
gyroscope_df.index = pd.to_datetime(gyroscope_df["epoch (ms)"], unit="ms")

""" 
    DELETING unnecessary columns from Dataframes
    Note: We are deleting the COLUMN "epoch (ms)" (VALUE), not the Index Column (KEY)
"""
del accelerometer_df["epoch (ms)"]
del accelerometer_df["time (01:00)"]
del accelerometer_df["elapsed (s)"]

del gyroscope_df["epoch (ms)"]
del gyroscope_df["time (01:00)"]
del gyroscope_df["elapsed (s)"]

# NOW we have a clean dataframe with the timestamp as the index [epoch (ms)]

"""
--------------------------------------------------------------
    Turn into function
--------------------------------------------------------------
"""

# This is where we clean up the code and functions
# Note this focus on the CSV files, changes will be made to reflect what file is used. 

""" 
    Note, I changed the following variables to easy code
        filesCSV => files
        accelerometer_df => acc_df
        gyroscope_df => gyr_df
        accelerometer_set => acc_set
        gyroscope_set => gyr_set
        x => f
"""


files = glob("../../data/raw/MetaMotion/*")

def read_data_from_files (files):
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    acc_set = 1
    gyr_set = 1

    for f in files:
        participant = f.split("-")[0].replace(data_path, "")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("1234").rstrip("_MetaWear_2019").rstrip("rpe6").rstrip("rpe7").rstrip("rpe8").rstrip("rpe9")
    
        df = pd.read_csv(f)
    
        df["participant"] = participant
        df["label"] = label
        df["category"] = category

        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df, df])
    
        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df, df])
            
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")
            
    del acc_df["epoch (ms)"]
    del acc_df["time (01:00)"]
    del acc_df["elapsed (s)"]

    del gyr_df["epoch (ms)"]
    del gyr_df["time (01:00)"]
    del gyr_df["elapsed (s)"]
    
    return acc_df, gyr_df

acc_df, gyr_df = read_data_from_files (files)


gyr_df.columns = [
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]

acc_df.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "participant",
    "label",
    "category",
    "set",
]
   
merged_data_Complete2 = gyr_df.combine_first(acc_df)

merged_data_Complete2.columns

merged_data_Complete2.head(5000)

Complete_Data_Set_Merged = merged_data_Complete2.reindex(columns=["acc_x", "acc_y","acc_z", "gyr_x", "gyr_y", "gyr_z", "participant", "label", "category", "set"] )

merged_data_Complete2.columns

""" 
--------------------------------------------------------------
    Merging datasets
--------------------------------------------------------------
 """

""" 
    .iloc is used to select columns when merging data. So same columns to avoid redundent columns
    Note when AXIS=1, the data is merge by COLUMNS.
    If AXIS=0, the data would merge by ROWS
"""

# Note if we need to correct the data, this is where this should happen. 

""" 
Tried this way and it is over complex for what we need. 

data_merged_Acc_Lead = pd.concat([acc_df.iloc[:,:3], gyr_df], axis=1)
data_merged_Acc_Lead.info()

data_merged_Gyr_Lead = pd.concat([gyr_df.iloc[:,:3], acc_df], axis=1)


data_merge_cleaned_Acc = data_merged_Acc_Lead.dropna(subset=["participant"])
data_merge_cleaned_Acc.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]

data_merge_cleaned_Gyr = data_merged_Gyr_Lead.dropna(subset=["participant"])

data_merge_cleaned_Gyr.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]

merged_data_Complete = data_merge_cleaned_Acc.combine_first(data_merge_cleaned_Gyr)

merged_data_Complete.head(20650)
 
merged_data_Complete.info()
"""

"""
This just drops rows that show any NaN values. This allows us to see how many rows share values with ACC and GYR. Remember, the GYR data was taken at a higher interval rate than the ACC data. 
    data_merged.dropna()
"""

"""
--------------------------------------------------------------
    Resample data (frequency conversion)
--------------------------------------------------------------
"""


# Accelerometer:    12.500HZ == 1/12.5 sec
# Gyroscope:        25.000Hz == 1/25 sec

sampling = {
    'acc_x': 'mean',
    'acc_y': 'mean',
    'acc_z': 'mean',
    'gyr_x': 'mean',
    'gyr_y': 'mean',
    'gyr_z': 'mean',
    'label': 'last',
    'category': 'last',
    'participant': 'last',
    'set': 'last'
}

#merged_data_Complete.columns

Complete_Data_Set_Merged[:1000].resample(rule="200ms").apply(sampling)

# Split by Day
days = [g for n, g in Complete_Data_Set_Merged.groupby(pd.Grouper(freq="D"))]

data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])

data_resampled.info()

data_resampled["set"] =  data_resampled["set"].astype("int")

data_resampled_Completed = data_resampled.reindex(columns=["acc_x", "acc_y","acc_z", "gyr_x", "gyr_y", "gyr_z", "participant", "label", "category", "set"])

"""
--------------------------------------------------------------
    Export dataset
--------------------------------------------------------------
"""

data_resampled_Completed.to_pickle("../../data/interim/01_data_processed.pkl")