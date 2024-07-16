import subprocess
from preprocessing import Event_data
import numpy as np
import pandas as pd

def download_with_wget(url, output_directory='.'):
    # Construct the wget command
    command = ['wget', url, '-P', output_directory]

    # Execute the command
    try:
        subprocess.run(command, check=True)
        print(f"Downloaded {url} successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {url}. Error: {e}")

if __name__ == "__main__":
    #path for the wyscout data
    event_path='./event'
    matches_path='./matches'

    #download the wyscout data
    event_url = "https://figshare.com/ndownloader/files/14464685/events.zip"
    matches_url = "https://figshare.com/ndownloader/files/14464622/matches.zip"
    download_with_wget(event_url, event_path)
    download_with_wget(matches_url, matches_path)

    #unzip the downloaded files
    subprocess.run(['unzip', 'event/events.zip', '-d', 'event'])
    subprocess.run(['unzip', 'matches/matches.zip', '-d', 'matches'])

    #remove the unnecessary files (expect England/France/Italy/Spain/Germany files)
    subprocess.run(['rm', '-rf', 'event/events.zip'])
    subprocess.run(['rm', '-rf', 'matches/matches.zip'])
    subprocess.run(['rm', '-rf', 'event/events_European_Championship.json'])
    subprocess.run(['rm', '-rf', 'event/events_World_Cup.json'])
    subprocess.run(['rm', '-rf', 'matches/matches_European_Championship.json'])
    subprocess.run(['rm', '-rf', 'matches/matches_World_Cup.json'])

    #load and preprocess the data (increase max_workers for faster processing)
    wyscout_df=Event_data(data_provider='wyscout',event_path=event_path,wyscout_matches_path=matches_path,
                          preprocess_method="NMSTPP",max_workers=1).preprocessing()
    wyscout_df.to_csv('data.csv',index=False)

    #split the data into train valid and test
    Train_ratio=0.8 
    Valid_ratio=0.1 
    Test_ratio=0.1 

    Train_id=[]
    Valid_id=[]
    Test_id=[]
    for i in np.unique(wyscout_df[['comp']]):
        temp=wyscout_df[wyscout_df['comp']==i]
        id_list=temp.match_id.unique()
        Train_id+=id_list[0:round(temp.match_id.nunique()*Train_ratio)].tolist()
        Valid_id+=id_list[round(temp.match_id.nunique()*Train_ratio):round(temp.match_id.nunique()*(Train_ratio+Valid_ratio))].tolist()
        Test_id+=id_list[round(temp.match_id.nunique()*(Train_ratio+Valid_ratio)):].tolist()

    train=wyscout_df[wyscout_df["match_id"].isin(Train_id)]
    valid=wyscout_df[wyscout_df["match_id"].isin(Valid_id)]
    test=wyscout_df[wyscout_df["match_id"].isin(Test_id)]

    train.to_csv("train.csv",index=False)
    valid.to_csv("valid.csv",index=False)
    test.to_csv("test.csv",index=False)
    print("---------------done-----------------")

