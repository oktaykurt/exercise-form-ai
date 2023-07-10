import pandas as pd
import os

# folder containing csv files
folder = "pose-coordinates"
# initialize an empty list to store dataframes
dfs = []

# loop over all csv files
for filename in os.listdir(folder):
    if filename.endswith('.csv'):
        # read csv file into dataframe
        df = pd.read_csv(os.path.join(folder, filename))
        
        # label dataframe based on file name
        if "lunge" in filename:
            df['exercise'] = 'lunge'
        elif "squat" in filename:
            df['exercise'] = 'squat'
        
        # append dataframe to list
        dfs.append(df)

# concatenate all dataframes into one
final_df = pd.concat(dfs)

# write final dataframe to csv
final_df.to_csv('combined_exercise_data.csv', index=False)
