import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display


""" 
--------------------------------------------------------------
    Load data
--------------------------------------------------------------
""" 

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")


"""
--------------------------------------------------------------
    Plot single columns
--------------------------------------------------------------
"""

set_df = df[df["set"] ==1]
"""
    NOTE: We had to limit our DF to a set ( SET_DF ), to limit errors. If we the data is confusing if we try to run all the sets together. BUT that makes sense in a sense of what we are trying to collect in reference to data for this project. This is not of project that the data is continous timeframe. Instead this is a project where each set is equal in value. The time frame is only long because we had to collect mutliple sets over the course of a week.
"""

plt.plot(set_df["acc_y"])
""" 
^^^ NOTE: That INDEX column is set at the X-AXIS by default. In this data set that means, EPOCH(MS) == X-AXIS VALUE if we use PLT.PLIT() command. 
    NOTE: We did not have to use INDEX-COLUMN value to declare ACC_Y, we just had to stay the proper COLUMN-NAME
"""

plt.plot(set_df["acc_y"].reset_index(drop=True))
"""
    NOTE: This just drops the TIMESTAMP-VALUE and sets INDEX to just a value of 0 to N (N represents value after the index is COUNTED)
""" 
"""
    Notes: This codes does the same thing as the code below
for x in df["label"].unique():
    subset = df[df["label"] == x]
    # display(subset.head(2))
    fig, ax = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True), label=x)
    plt.legend()
    plt.show()
"""



"""
--------------------------------------------------------------
    Plot all exercises
--------------------------------------------------------------
"""
    
for label in df["label"].unique():
    subset = df[df["label"] == label]
    # display(subset.head(2))
    fig, ax = plt.subplots()
    # plt.figure().add_subplot().axis([0, 50,-2, 2])
    plt.plot(subset[0:100]["acc_y"].reset_index(drop=True), label=label)
    plt.legend()
    plt.show()


"""
--------------------------------------------------------------
    Adjust plot settings
--------------------------------------------------------------
"""

mpl.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams["figure.figsize"] = (20 , 5)
mpl.rcParams["figure.dpi"] = 75
mpl.rcParams["font.sans-serif"] = 'Arial'


"""
--------------------------------------------------------------
    Compare medium vs. heavy sets
--------------------------------------------------------------
"""

category_df = df.query("label == 'squat'").query("participant == 'A'").reset_index()
category_df = df.query("label == 'bench'").query("participant == 'A'").reset_index()
category_df = df.query("label == 'ohp'").query("participant == 'A'").reset_index()
category_df = df.query("label == 'dead'").query("participant == 'A'").reset_index()
category_df = df.query("label == 'row'").query("participant == 'A'").reset_index()
category_df = df.query("label == 'rest'").query("participant == 'A'").reset_index()


fig,ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()

"""
--------------------------------------------------------------
    Compare participants
--------------------------------------------------------------
"""

participant_df = df.query("label == 'bench'").sort_values("participant").reset_index()

fig,ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()


"""
--------------------------------------------------------------
    Plot multiple axis
--------------------------------------------------------------
"""

label = "squat"
participant = "A"
all_axis_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()

fig, ax = plt.subplots()
all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
ax.set_ylabel("y-axis")
ax.set_xlabel("samples")
plt.legend()


"""
--------------------------------------------------------------
    Create a loop to plot all combinations per sensor
--------------------------------------------------------------
"""
mpl.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams["figure.figsize"] = (20 , 5)
mpl.rcParams["figure.dpi"] = 75
mpl.rcParams["font.sans-serif"] = 'Arial'


labels = df["label"].unique()
participants = df["participant"].unique()

# Plot Acceleration Graphs
for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}' ")
            .query(f"participant == '{participant}' ")
            .reset_index()
        )
        
        # Use the following if statement to limit empty data frame. For here, its when a participant did not do an exercise.
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            ax.set_ylabel("y-axis")
            ax.set_xlabel("Samples")
            plt.title(f"{label} ({participant})".title())
            plt.legend()
 
 
# Plot Gyroscope Graphs            
for label in labels:
    for participant in participants:
        all_axis_df = (
            df.query(f"label == '{label}' ")
            .query(f"participant == '{participant}' ")
            .reset_index()
        )
        
        # Use the following if statement to limit empty data frame. For here, its when a participant did not do an exercise.
        if len(all_axis_df) > 0:
            fig, ax = plt.subplots()
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            ax.set_ylabel("y-axis")
            ax.set_xlabel("Samples")
            plt.title(f"{label} ({participant})".title())
            plt.legend()


"""
--------------------------------------------------------------
    Combine plots in one figure
--------------------------------------------------------------
"""
label = "row"
participant = "A"
combined_plot_df = (
    df.query(f"label == '{label}' ")
    .query(f"participant == '{participant}' ")
    .reset_index(drop=True)
)

fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
ax[1].set_xlabel("samples")

"""
--------------------------------------------------------------
    Loop over all combinations and export for both sensors
--------------------------------------------------------------
"""

labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        combined_plot_df = (
            df.query(f"label == '{label}' ")
            .query(f"participant == '{participant}' ")
            .reset_index()
        )

        if len(combined_plot_df) > 0:
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

            ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            ax[1].set_xlabel("samples")
            plt.savefig(f"../../reports/figures/{label.title()}-{participant}.png")            
            plt.show()
            

 