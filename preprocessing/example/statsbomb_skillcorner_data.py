import os
import pandas as pd
from statsbombpy import sb
from tqdm import tqdm
import json
import pdb
import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import sports.event_data.load_data as load_data

#Statsbomb API
creds = {"user": "input your Statsbomb api user name here", "passwd": "input your Statsbomb api password here"}
#Statsbomb event data saving dir
save_dir = "/statsbomb/events"
#path to the skillcorner tracking data
tracking_path="/skillcorner/tracking"
#path to the skillcorner match data
match_id_path="/skillcorner/match"

os.makedirs(save_dir, exist_ok=True)

def convert_df_in_dict(d):
    for key, value in d.items():
        if isinstance(value, pd.DataFrame):
            d[key] = value.to_dict(orient='records')
        elif isinstance(value, dict):
            convert_df_in_dict(value)
    return d
# Get Statsbomb matches data
matches = sb.matches(competition_id=11, season_id=281, creds=creds)
matches["competition_id"] = 11
matches["season_id"] = 281
#moev the competition_id and season_id to the first column
cols = matches.columns.tolist()
cols = cols[-2:] + cols[:-2]
matches = matches[cols]
#save the matches to csv
matches.to_csv(os.path.join(save_dir, "matches.csv"), index=False)


# Get Statsbomb lineups and events
os.makedirs(os.path.join(save_dir, "lineups"), exist_ok=True)
os.makedirs(os.path.join(save_dir, "events"), exist_ok=True)
for match_id in tqdm(matches["match_id"].unique()):
    lineups = sb.lineups(match_id=match_id, creds=creds)
    events = sb.events(match_id=match_id, include_360_metrics=True, creds=creds)
    events.to_csv(os.path.join(save_dir, "events", f"{match_id}.csv"), index=False)
    #save the lineups as json and with row changes
    lineups = convert_df_in_dict(lineups)
    with open(os.path.join(save_dir, "lineups", f"{match_id}.json"), "w") as f:
        json.dump(lineups, f, indent=4)


#load_statsbomb_skillcorner
statsbomb_match_id=3894907 #for the match id matching refer to id_matching.csv
skillcorner_match_id=1553748
output_dir="data.csv"

statsbomb_skillcorner_event_path=save_dir+'/events'
statsbomb_skillcorner_tracking_path=tracking_path
statsbomb_skillcorner_match_path=match_id_path
statsbomb_skillcorner_df=load_data.load_statsbomb_skillcorner(statsbomb_skillcorner_event_path,statsbomb_skillcorner_tracking_path,
                                                    statsbomb_skillcorner_match_path,statsbomb_match_id,skillcorner_match_id)
statsbomb_skillcorner_df.to_csv(output_dir,index=False)



