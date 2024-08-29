import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as patches
import os
import pdb

FIELD_LENGTH = 105.0  # unit: meters
FIELD_WIDTH = 68.0  # unit: meters
GOAL_WIDTH = 7.32  # unit: meters
PENALTY_X = 105.0/2-16.5 # left point (unit: meters)
PENALTY_Y = 40.32 # upper point (unit: meters)


def plot_row_soccer(df, row, save_path):
    if not isinstance(df, pd.DataFrame):
        if isinstance(df, str):
            df = pd.read_csv(df)
        else:
            raise ValueError("The input is not a dataframe or a path to a csv file")
        
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(bottom=0.2)

    # Flip the y-axis
    ax.invert_yaxis()

    # Plot the pitch

    # Center line
    ax.plot([FIELD_LENGTH/2, FIELD_LENGTH/2], [0, FIELD_WIDTH], color="black", linewidth=0.7)

    # Penalty areas
    # pdb.set_trace()
    ax.plot([PENALTY_X+FIELD_LENGTH/2, FIELD_LENGTH], [(FIELD_WIDTH-PENALTY_Y)/2, (FIELD_WIDTH-PENALTY_Y)/2], color="black", linewidth=0.7)
    ax.plot([PENALTY_X+FIELD_LENGTH/2, FIELD_LENGTH], [(FIELD_WIDTH+PENALTY_Y)/2, (FIELD_WIDTH+PENALTY_Y)/2], color="black", linewidth=0.7)
    ax.plot([PENALTY_X+FIELD_LENGTH/2, PENALTY_X+FIELD_LENGTH/2,], [(FIELD_WIDTH-PENALTY_Y)/2, (FIELD_WIDTH+PENALTY_Y)/2], color="black", linewidth=0.7)

    ax.plot([FIELD_LENGTH/2-PENALTY_X, 0], [(FIELD_WIDTH-PENALTY_Y)/2, (FIELD_WIDTH-PENALTY_Y)/2], color="black", linewidth=0.7)
    ax.plot([FIELD_LENGTH/2-PENALTY_X, 0], [(FIELD_WIDTH+PENALTY_Y)/2, (FIELD_WIDTH+PENALTY_Y)/2], color="black", linewidth=0.7)
    ax.plot([FIELD_LENGTH/2-PENALTY_X, FIELD_LENGTH/2-PENALTY_X], [(FIELD_WIDTH-PENALTY_Y)/2, (FIELD_WIDTH+PENALTY_Y)/2], color="black", linewidth=0.7)

    # Goal areas
    ax.plot([5.5, 0], [(FIELD_WIDTH-18.32)/2, (FIELD_WIDTH-18.32)/2], color="black", linewidth=0.7)
    ax.plot([5.5, 0], [(FIELD_WIDTH+18.32)/2, (FIELD_WIDTH+18.32)/2], color="black", linewidth=0.7)
    ax.plot([5.5, 5.5], [(FIELD_WIDTH-18.32)/2, (FIELD_WIDTH+18.32)/2], color="black", linewidth=0.7)

    ax.plot([FIELD_LENGTH-5.5, FIELD_LENGTH], [(FIELD_WIDTH-18.32)/2, (FIELD_WIDTH-18.32)/2], color="black", linewidth=0.7)
    ax.plot([FIELD_LENGTH-5.5, FIELD_LENGTH], [(FIELD_WIDTH+18.32)/2, (FIELD_WIDTH+18.32)/2], color="black", linewidth=0.7)
    ax.plot([FIELD_LENGTH-5.5, FIELD_LENGTH-5.5], [(FIELD_WIDTH-18.32)/2, (FIELD_WIDTH+18.32)/2], color="black", linewidth=0.7)

    # # Goals
    # ax.plot([-2, -2], [(FIELD_WIDTH-GOAL_WIDTH)/2, (FIELD_WIDTH+GOAL_WIDTH)/2], color="black", linewidth=10)
    # ax.plot([FIELD_LENGTH+2, FIELD_LENGTH+2], [(FIELD_WIDTH-GOAL_WIDTH)/2, (FIELD_WIDTH+GOAL_WIDTH)/2], color="black", linewidth=10)

    # Field outline
    ax.plot([0, FIELD_LENGTH], [0, 0], color="black", linewidth=2)
    ax.plot([0, FIELD_LENGTH], [FIELD_WIDTH, FIELD_WIDTH], color="black", linewidth=2)
    ax.plot([0, 0], [0, FIELD_WIDTH], color="black", linewidth=2)
    ax.plot([FIELD_LENGTH, FIELD_LENGTH], [0, FIELD_WIDTH], color="black", linewidth=2)

    # Center circle
    c = patches.Circle(xy=(FIELD_LENGTH/2, FIELD_WIDTH/2), radius=9.15, fill=False, ec='black', linewidth=0.7)
    ax.add_patch(c)

    # Penalty arcs
    a = patches.Arc((11, FIELD_WIDTH/2), 9.15*2, 9.15*2, theta1=270+37, theta2=90-37, linewidth=0.7)
    ax.add_patch(a)
    a = patches.Arc((FIELD_LENGTH-11, FIELD_WIDTH/2), 9.15*2, 9.15*2, theta1=90+36, theta2=270-36, linewidth=0.7)
    # a = patches.Arc((-FIELD_LENGTH / 2 + 11, 0), 9.15*2, 9.15*2, theta1=270+34, theta2=90-34, linewidth=0.7)
    ax.add_patch(a)

    # Set axis limits
    ax.set_xlim(-5, FIELD_LENGTH+5)
    ax.set_ylim(FIELD_WIDTH+5, -5)

    # Plot the player positions
    df = df.reset_index(drop=True)
    row_df = df.iloc[row]
    if row_df.home_team == 1:
        for i in range(1, 24):
            x = row_df[f"h{i}_x"]+FIELD_LENGTH/2
            y = -(row_df[f"h{i}_y"]+FIELD_WIDTH/2 - FIELD_WIDTH/2)+FIELD_WIDTH/2
            if x == 0 and y == 0:
                continue
            ax.plot(x, y, 'o', color='red')
        for i in range(1, 24):
            x = row_df[f"a{i}_x"]+FIELD_LENGTH/2
            y = -(row_df[f"a{i}_y"]+FIELD_WIDTH/2 - FIELD_WIDTH/2)+FIELD_WIDTH/2
            if x == 0 and y == 0:
                continue
            ax.plot(x, y, 'o', color='blue')
    else:
        for i in range(1, 24):
            x = -(row_df[f"h{i}_x"]+FIELD_LENGTH/2 - FIELD_LENGTH/2)+FIELD_LENGTH/2
            y = -(row_df[f"h{i}_y"]+FIELD_WIDTH/2 - FIELD_WIDTH/2)+FIELD_WIDTH/2
            if x == 0 and y == 0:
                continue
            ax.plot(x, y, 'o', color='red')
        for i in range(1, 24):
            x = -(row_df[f"a{i}_x"]+FIELD_LENGTH/2 - FIELD_LENGTH/2)+FIELD_LENGTH/2
            y = -(row_df[f"a{i}_y"]+FIELD_WIDTH/2 - FIELD_WIDTH/2)+FIELD_WIDTH/2
            if x == 0 and y == 0:
                continue
            ax.plot(x, y, 'o', color='blue')

    
    #plot the event location
    x = row_df["start_x"]
    y = row_df["start_y"]
    ax.plot(x, y, 'o', color='black', markersize=3)
     
    # Set the figure title
    ax.set_title(f"Row {row}, action: {row_df['action']}, seconds: {row_df['seconds']}\n red: home team, blue: away team, black: event location")

    # Save the plot
    plt.savefig(save_path + f"/row_{row}.png")
    plt.close(fig)
