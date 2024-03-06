import pandas as pd
from glob import glob

# List ALL files
filesAll = glob("../../data/raw/MetaMotion/*")

# List ONLY file with CVS as an extension
filesCSV = glob("../../data/raw/MetaMotion/*.csv")
len(filesCSV)

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

accelerometer_df = pd.DataFrame()
gyroscope_df = pd.DataFrame()

accelerometer_set = 1
gyroscope_set = 1


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
    "gyr_x", "gyr_y", "gyr_z", "participant", "label", "category", "set",
]

acc_df.columns = [
    "acc_x", "acc_y", "acc_z", "participant", "label", "category", "set",
]
   
merged_data_Complete2 = gyr_df.combine_first(acc_df)

Complete_Data_Set_Merged = merged_data_Complete2.reindex(columns=["acc_x", "acc_y","acc_z", "gyr_x", "gyr_y", "gyr_z", "participant", "label", "category", "set"] )


# Accelerometer:    12.500HZ == 1/12.5 sec
# Gyroscope:        25.000Hz == 1/25 sec

sampling = {
    'acc_x': 'mean', 'acc_y': 'mean', 'acc_z': 'mean', 'gyr_x': 'mean', 'gyr_y': 'mean', 'gyr_z': 'mean', 'label': 'last', 'category': 'last', 'participant': 'last', 'set': 'last'
}


days = [g for n, g in Complete_Data_Set_Merged.groupby(pd.Grouper(freq="D"))]

data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])

data_resampled["set"] =  data_resampled["set"].astype("int")

data_resampled_Completed = data_resampled.reindex(columns=["acc_x", "acc_y","acc_z", "gyr_x", "gyr_y", "gyr_z", "participant", "label", "category", "set"])

data_resampled_Completed.to_pickle("../../data/interim/01_data_processed.pkl")