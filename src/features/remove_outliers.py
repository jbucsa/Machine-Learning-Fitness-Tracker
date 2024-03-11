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

def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """ Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    )

    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()


# --------------------------------------------------------------
# Interquartile range (distribution based)
# --------------------------------------------------------------

"""
IQR is instrumental in creating boxplots, a vital tool for visually analyzing data distribution.
The box in a boxplot represents the IQR, and the line within the box indicates the median (Q2).
The "whiskers" extending from the box represent the range of data points within 1.5 times the IQR. Points falling outside this range are considered potential outliers.
"""

# Insert IQR function

def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset


# Plot a single column

col = "acc_x"
dataset = mark_outliers_iqr(df, col)

plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+"_outlier", reset_index=True)


# Loop over all columns

""" 
    NOTE: This is the same code.
for x in outlier_columns:
    dataset = mark_outliers_iqr(df, x)
    plot_binary_outliers(dataset=dataset, col=x, outlier_col=x+"_outlier", reset_index=True)
"""

for col in outlier_columns:
    dataset = mark_outliers_iqr(df, col)
    plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+"_outlier", reset_index=True)


# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------

""" 
Chauvenets criteron (distribution based)

Chauvenet's criterion is a specific method for detecting outliers in data. It is based on the idea that, for a given dataset, the probability of an outlier occurring is relatively low. Chauvenet's criterion can be useful in certain situations, such as when you want to identify outliers in data that follows a normal distribution. However, it is not necessarily the best method for detecting outliers in all cases, and there are other methods that may be more appropriate in certain situations.

According to Chauvenet’s criterion we reject a measurement (outlier) from a dataset of size N when it’s probability of observation is less than 1/2N. A generalization is to replace the value 2 with a parameter C.

Normal distribution
It's important to note that Chauvenet's criterion is only applicable to datasets that are normally distributed. If your dataset is not normally distributed, this method may not be suitable for identifying outliers.


"""

# Check for normal distribution
# .plot.hist runs a bar chack that then can be obversed to see if the data fits a BELL SHAPED CURVE. If YES, then the data is normally distributed.

df[outlier_columns[0:3] + ["label"]].plot.hist(by="label", figsize=(20,20), layout=(3, 3))

df[outlier_columns[3:6] + ["label"]].plot.hist(by="label", figsize=(20,20), layout=(3, 3))

# Insert Chauvenet's function

def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.
    
    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption 
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset

# Loop over all columns

for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df, col)
    plot_binary_outliers(dataset=dataset, col=col, outlier_col=col+"_outlier", reset_index=True)

# --------------------------------------------------------------
# Local outlier factor (distance based)
# --------------------------------------------------------------

""" 
Unsupervised Outlier Detection using the Local Outlier Factor (LOF).

The anomaly score of each sample is called the Local Outlier Factor. It measures the local deviation of the density of a given sample with respect to its neighbors. It is local in that the anomaly score depends on how isolated the object is with respect to the surrounding neighborhood. More precisely, locality is given by k-nearest neighbors, whose distance is used to estimate the local density. By comparing the local density of a sample to the local densities of its neighbors, one can identify samples that have a substantially lower density than their neighbors. These are considered outliers.

"""

# Insert LOF function

def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.
    
    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """
    
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores

# Loop over all columns
dataset, outliers, X_scores = mark_outliers_lof(df, outlier_columns)

for col in outlier_columns:
    plot_binary_outliers(dataset=dataset, col=col, outlier_col="outlier_lof", reset_index=True)

# --------------------------------------------------------------
# Check outliers grouped by label
# --------------------------------------------------------------

label = "bench"
for col in outlier_columns:
    dataset = mark_outliers_iqr(df[df["label"] == label], col)
    plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)
    
for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df[df["label"] == label], col)
    plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)
    
dataset, outliers, X_scores = mark_outliers_lof(df[df["label"] == label], outlier_columns)
for col in outlier_columns:
    plot_binary_outliers(dataset=dataset, col=col, outlier_col="outlier_lof", reset_index=True)    



label = "squat"
for col in outlier_columns:
    dataset = mark_outliers_iqr(df[df["label"] == label], col)
    plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)
    
for col in outlier_columns:
    dataset = mark_outliers_chauvenet(df[df["label"] == label], col)
    plot_binary_outliers(dataset, col, col + "_outlier", reset_index=True)
    
dataset, outliers, X_scores = mark_outliers_lof(df[df["label"] == label], outlier_columns)
for col in outlier_columns:
    plot_binary_outliers(dataset=dataset, col=col, outlier_col="outlier_lof", reset_index=True)  


# --------------------------------------------------------------
# Choose method and deal with outliers
# --------------------------------------------------------------

# Test on single column
col = "gyr_z"
dataset = mark_outliers_chauvenet(df, col=col)
dataset[dataset["gyr_z_outlier"]]

# .loc[dataset["gyr_z_outlier"], "gyr_z"] this takes all the values in "gyr_z" column that share a row where the "gyr_z_outlier" is equal to 'TRUE' and sets the "gyr_z" value to "NaN"
dataset.loc[dataset["gyr_z_outlier"], "gyr_z"] = np.nan


# Create a loop

outliers_removed_df = df.copy()
for col in outlier_columns:
    for label in df["label"].unique():
        dataset = mark_outliers_chauvenet(df[df["label"] == label], col)
        
        # Replace the VALUES marked as OUTLIERS with NaN
        dataset.loc[dataset[col + "_outlier"], col]= np.nan
        
        # Update the COLUMN in the ORIGINAL DATAFRAME
        outliers_removed_df.loc[(outliers_removed_df["label"] == label), col]= dataset[col]
        
        n_outliers = len(dataset) - len(dataset[col].dropna())
        print(f"Removed {n_outliers} from {col} for {label}")


outliers_removed_df.info()

# --------------------------------------------------------------
# Export new dataframe
# --------------------------------------------------------------

outliers_removed_df.to_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")


outliers_removed_df_completed = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenets.pkl")
