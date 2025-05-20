import os
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat
import numpy as np
import pandas as pd
import gc
import gdown
import zipfile
import os

class Basketball_space_data:
    def __init__(self,data_provider,data_path, *args, **kwargs):
        self.data_provider = data_provider
        self.data_path = data_path


    def preprocessing(self,nb_process_game):

        if self.data_provider == "SportVU_NBA":

            # Read sequence info
            sequence_info = pd.read_csv(f"{self.data_path}/nba_datalength_updated.csv")
            sequence_info = sequence_info.rename(columns={'eventid': 'attackid',
                                                          'eventid_original': 'attackid_original'})

            event_data = loadmat(f'{self.data_path}/allevents_dataset.mat')['event'][0]

            # Instead of keeping all data in memory, write to CSV immediately in chunks
            with open(f'{self.data_path}/reshape_data.csv', 'w') as f_data:
                # Get column names for the main dataframe
                sample_columns = list(sequence_info.columns) + [
                    'f_id', 'event_label', 
                    'calc_fid', 'last_choice', 'calc_posx', 'calc_posy'
                ] + [
                    'x_att0','y_att0','x_att1','y_att1','x_att2','y_att2','x_att3','y_att3','x_att4','y_att4',
                    'x_def0','y_def0','x_def1','y_def1','x_def2','y_def2','x_def3','y_def3','x_def4','y_def4',
                    'x_ball','y_ball','z_ball',
                    'vx_att0','vy_att0','vx_att1','vy_att1','vx_att2','vy_att2','vx_att3','vy_att3','vx_att4','vy_att4',
                    'vx_def0','vy_def0','vx_def1','vy_def1','vx_def2','vy_def2','vx_def3','vy_def3','vx_def4','vy_def4',
                    'vx_ball','vy_ball','vz_ball',
                    'clock','shot_clock',
                    'ID_att0','ID_att1','ID_att2','ID_att3','ID_att4','ID_def0','ID_def1','ID_def2','ID_def3','ID_def4',
                    'jersey_att0','jersey_att1','jersey_att2','jersey_att3','jersey_att4',
                    'jersey_def0','jersey_def1','jersey_def2','jersey_def3','jersey_def4',
                    'ball_hold','ball_holder'
                ]
                
                f_data.write(','.join(sample_columns) + '\n')
                
                # Process each game
                if nb_process_game == 'ALL':
                    length_reshape_game = sum(1 for f in os.scandir(f"{self.data_path}/modified_scoreDataset") if f.is_file())
                
                else:
                    length_reshape_game = nb_process_game

                for game_id in tqdm(range(0, length_reshape_game)):
                    
                    game_str = str(game_id+1).zfill(3)
                    game_sequences = sequence_info[sequence_info['game'] == game_id+1]
                    nb_sequences = len(game_sequences)
                    
                    
                    tracking_data = loadmat(f"{self.data_path}/modified_scoreDataset/attackDataset_game{game_str}.mat")['data'][0]
                    
                        
                    # Process each sequence in the game
                    for j in range(nb_sequences):
                        # Get sequence info
                        current_sequence = game_sequences.iloc[j].to_dict()
                        
                        # Get event data for this sequence
                        event_df = pd.DataFrame(event_data[game_id][0][j])
                        
                        # Check if event_df only has 6 columns mean incomplete data
                        if event_df.shape[1] == 6:
                            continue
                        
                        event_df.columns = ['f_id', 'event_label', 'score', 'ball_x', 'ball_y', 
                                            'ball_holder_pid_idx', 'calc_fid', 'last_choice', 'calc_posx', 'calc_posy']
                        
                        col_to_int= ['f_id','event_label','score','calc_fid','last_choice']
                        event_df[col_to_int] = event_df[col_to_int].astype(int)

                        
                        # Delete unnecessary ball-related columns
                        if 'ball_x' in event_df.columns and 'ball_y' in event_df.columns and 'ball_holder_pid_idx' in event_df.columns and 'score' in event_df.columns:
                            event_df = event_df.drop(['ball_x', 'ball_y', 'ball_holder_pid_idx','score'], axis=1)
                        
                       

                        # Get tracking data for this sequence
                        tracking_df = pd.DataFrame(tracking_data[j])
                        
                        tracking_df.columns = [
                            'x_att0','y_att0','x_att1','y_att1','x_att2','y_att2','x_att3','y_att3','x_att4','y_att4',
                            'x_def0','y_def0','x_def1','y_def1','x_def2','y_def2','x_def3','y_def3','x_def4','y_def4',
                            'x_ball','y_ball','z_ball',
                            'vx_att0','vy_att0','vx_att1','vy_att1','vx_att2','vy_att2','vx_att3','vy_att3','vx_att4','vy_att4',
                            'vx_def0','vy_def0','vx_def1','vy_def1','vx_def2','vy_def2','vx_def3','vy_def3','vx_def4','vy_def4',
                            'vx_ball','vy_ball','vz_ball',
                            'clock','shot_clock',
                            'ID_att0','ID_att1','ID_att2','ID_att3','ID_att4','ID_def0','ID_def1','ID_def2','ID_def3','ID_def4',
                            'jersey_att0','jersey_att1','jersey_att2','jersey_att3','jersey_att4',
                            'jersey_def0','jersey_def1','jersey_def2','jersey_def3','jersey_def4',
                            'ball_hold','ball_holder'
                        ]

                        # Conversion en int
                        cols_to_int = [
                            'ID_att0','ID_att1','ID_att2','ID_att3','ID_att4','ID_def0','ID_def1','ID_def2','ID_def3','ID_def4',
                            'jersey_att0','jersey_att1','jersey_att2','jersey_att3','jersey_att4',
                            'jersey_def0','jersey_def1','jersey_def2','jersey_def3','jersey_def4',
                            'ball_hold','ball_holder'
                        ]
                        tracking_df[cols_to_int] = tracking_df[cols_to_int].astype(int)
                        
                        # Check if the lengths match
                        if len(event_df) == len(tracking_df):
                            # Process row by row to avoid creating large DataFrames
                            for idx in range(len(event_df)):
                                row_data = []
                                # Add sequence info
                                for col in sequence_info.columns:
                                    row_data.append(str(current_sequence[col]))
                                # Add event data
                                for col_idx in range(event_df.shape[1]):
                                    row_data.append(str(event_df.iloc[idx, col_idx]))
                                # Add tracking data
                                for col_idx in range(tracking_df.shape[1]):
                                    row_data.append(str(tracking_df.iloc[idx, col_idx]))
                                
                                # Write row to file
                                f_data.write(','.join(row_data) + '\n')
                                    
                        # Clear memory
                        del event_df, tracking_df
                        gc.collect()
                    
                    # Clear game-specific memory
                    del tracking_data, game_sequences
                    gc.collect()
                    
            return pd.read_csv(f'{self.data_path}/reshape_data.csv')
    
    def download_data(self):
        id_allevents_dataset = "1GNaEO4C5kxJuJV2YHw6KpNDbmsAMXPqr"
        id_nba_datalength_updated = "19hUhLcE7gO_TVH6PjqAF6plWkh-mVCyU"
        id_basic_content_zip = "14y-U055lZ-D4snaaLKK1MVwwRc0Y4WS_"
        output_zip_path = f"{self.data_path}/basic_content.zip" 
        folder_to_extract_nested = "basic_content/modified_scoreDataset"
     
        gdown.download(f"https://drive.google.com/uc?id={id_allevents_dataset}", f"{self.data_path}/allevents_dataset.mat", quiet=False)
        print("downlaod allevents_dataset.mat ok")
        gdown.download(f"https://drive.google.com/uc?id={id_nba_datalength_updated}", f"{self.data_path}/nba_datalength_updated.csv", quiet=False)
        print("downlaod nba_datalength_updated.csv ok")
        gdown.download(f"https://drive.google.com/uc?id={id_basic_content_zip}", output_zip_path, quiet=False)
        print("downlaod basic_content.zip ok")
        print("extract modified_scoreDataset folder from the zip file")

        # Extract only the modified_scoreDataset folder from the zip file
        with zipfile.ZipFile(output_zip_path, 'r') as zip_ref:
            os.makedirs(f"{self.data_path}/modified_scoreDataset", exist_ok=True)
            
    
            for item in zip_ref.namelist():
                if item.startswith(folder_to_extract_nested):
                    filename = os.path.basename(item)
                    if filename and not item.endswith('/'):
                        source = zip_ref.open(item)
                        target = open(f"{self.data_path}/modified_scoreDataset/{filename}", "wb")
                        target.write(source.read())
                        target.close()
                        source.close()

        os.remove(output_zip_path)
        print("delete the unecessary zip file")
        return print("Data downloaded finish")