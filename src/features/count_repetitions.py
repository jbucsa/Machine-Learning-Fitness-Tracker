import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None

# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# Remove Rest Dataframe
df = df[df["label"] != "rest"]

# Calculate the Sum of Square for ACC and GYR and use np.sqrt() to find the square root of the resulting sum of the all squares. 
acc_r = np.sqrt((df["acc_x"] ** 2) + (df["acc_y"] ** 2) + (df["acc_z"] ** 2))
gyr_r = np.sqrt((df["gyr_x"] ** 2) + (df["gyr_y"] ** 2) + (df["gyr_z"] ** 2))

# Add a column to the df (DATADRAME) that is label for "acc_r" and "gyr_r"
df["acc_r"] = acc_r
df["gyr_r"] = gyr_r

# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------


# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------


# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------


# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------


# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------


# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------