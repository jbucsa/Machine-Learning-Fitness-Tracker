import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

""" 
This variable just calls out the first 6 columns in the data, these are the columns that hold our x, y, and z data. THEN we convert the INDEX to a LIST.

outlier_columns = df.columns[0:6] >> Index(['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z'], dtype='object')

outlier_columns = list(df.columns[0:6]) >> ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z'] 
"""

outlier_columns = list(df.columns[0:6])

# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------

plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"]= 100

# First step is to get a visualization of what is seen
df[["acc_x", "label"]].boxplot(by="label", figsize=(20,10))
df[["acc_y", "label"]].boxplot(by="label", figsize=(20,10))
df[["acc_z", "label"]].boxplot(by="label", figsize=(20,10))

df[["gyr_x", "label"]].boxplot(by="label", figsize=(20,10))
df[["gyr_y", "label"]].boxplot(by="label", figsize=(20,10))
df[["gyr_z", "label"]].boxplot(by="label", figsize=(20,10))


df[outlier_columns[0:3] + ["label"]].boxplot(by="label", figsize=(20,10), layout=(1, 3))

df[outlier_columns[3:6] + ["label"]].boxplot(by="label", figsize=(20,10), layout=(1, 3))


# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------

# Insert IQR function


# Plot a single column


# Loop over all columns


# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------

# Check for normal distribution


# Insert Chauvenet's function


# Loop over all columns


# --------------------------------------------------------------
# Local outlier factor (distance based)
# --------------------------------------------------------------

# Insert LOF function


# Loop over all columns


# --------------------------------------------------------------
# Check outliers grouped by label
# --------------------------------------------------------------


# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------

# Test on single column


# Create a loop

# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------