import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Read single CSV file
# --------------------------------------------------------------

single_file_accelerometer = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv"
    )

single_file_gyroscope = pd.read_csv(
    "../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv"
    )

# --------------------------------------------------------------
# List all data in data/raw/MetaMotion
# --------------------------------------------------------------

# List ALL files
filesAll = glob("../../data/raw/MetaMotion/*")

# List ONLY file with CVS as an extension
filesCSV = glob("../../data/raw/MetaMotion/*.csv")
len(filesCSV)

# --------------------------------------------------------------
# Extract features from filename
# --------------------------------------------------------------

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
# --------------------------------------------------------------
# Read all files
# --------------------------------------------------------------

accelerometer_df = pd.DataFrame()
gyroscope_df = pd.DataFrame()

accelerometer_set = 1
gyroscope_set = 1

for x in filesCSV:
    participant = x.split("-")[0].replace(data_path, "")
    label = x.split("-")[1]
    category = x.split("-")[2].rstrip("1234").rstrip("_MetaWear_2019")
    
    df = pd.read_csv(x)
    
    df["participant"] = participant
    df["label"] = label
    df["category"] = category

    


    
# for f_CVS in filesCSV:
#     participant = f_CVS.split("-")[0].replace(data_path, "")
#     label = f_CVS.split("-")[1]
#     category = f_CVS.split("-")[2].rstrip("1234").rstrip("_MetaWear_2019")

# --------------------------------------------------------------
# Working with datetimes
# --------------------------------------------------------------


# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------


# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------


# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz


# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------