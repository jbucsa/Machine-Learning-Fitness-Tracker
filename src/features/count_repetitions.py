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

# Separate the Dataframes from each other
bench_df = df[df["label"] == "bench"]
squat_df = df[df["label"] == "squat"]
row_df = df[df["label"] == "row"]
ohp_df = df[df["label"] == "ohp"]
bench_df = df[df["label"] == "bench"]
dead_df = df[df["label"] == "dead"]

# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------

plot_df = bench_df

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_r"].plot()

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_r"].plot()


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