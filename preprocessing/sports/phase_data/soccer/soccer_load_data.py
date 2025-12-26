#Target data provider [Metrica,Robocup 2D simulation,Statsbomb,Wyscout,Opta data,DataFactory,sportec]

import json
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import xml.etree.ElementTree as ET
# from statsbombpy import sb
import os
import pickle
from typing import List, Dict, Any

def load_bepro(tracking_xml_path: str, tracking_json_paths: list, event_path: str, meta_data_path: str) -> pd.DataFrame:
    """
    Loads and processes event and tracking data from soccer match recordings.

    This function combines event data with tracking data by merging based on event time. It also adds 
    additional features extracted from metadata, such as player information, and converts position 
    coordinates to the correct scale for analysis.

    Args:
        event_path (str): Path to the CSV file containing event data.
        tracking_path (str): Path to the XML file containing tracking data.
        meta_path (str): Path to the XML file containing match metadata (pitch, teams, players, etc.).
        verbose (bool, optional): If True, prints additional information about the merging process and 
                                  feature extraction. Default is False.

    Returns:
        pd.DataFrame: A DataFrame containing the merged and processed event and tracking data, 
                      with additional features including player positions, speeds, ball position, 
                      and metadata (e.g., player names, shirt numbers, positions).
    """

    def extract_tracking_data_from_xml(xml_path: str) -> List[Dict[str, Any]]:
        """
        Parse the XML file and extract tracking data for players and the ball.

        Args:
            xml_path (str): Path to the XML file.
        Returns:
            list of dict: A list containing tracking information for each player and the ball in each frame.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        tracking_data = []

        for frame in root.findall("frame"):
            frame_number = int(frame.get("frameNumber"))
            match_time = int(frame.get("matchTime"))
            
            # 処理対象の要素を <player> と <ball> の両方に拡張
            # findall("*") を使用することで、<frame> の直下にある全ての要素（player, ballなど）を取得
            for element in frame.findall("*"): 
                
                # タグ名に基づいて player_id と loc の属性名を設定
                if element.tag == "player":
                    player_id = element.get("playerId")
                    loc = element.get("loc")
                elif element.tag == "ball":
                    # ⭐ 変更点: <ball> タグの場合、player_id を "ball" とし、属性を取得
                    player_id = "ball"
                    loc = element.get("loc")
                else:
                    # 予期しないタグはスキップ
                    continue
                
                # loc 情報が存在しない場合はスキップ
                if loc is None:
                    continue

                # Convert loc string to float coordinates
                try:
                    # loc の形式は "[x, y]" を想定
                    x, y = map(float, loc.strip("[]").split(","))
                    
                    # 座標変換とデータ追加
                    tracking_data.append({
                        "frame": frame_number,
                        "match_time": match_time,
                        "player_id": player_id,
                        # 座標の正規化解除とフォーマット適用 (元のコードのロジックを維持)
                        "x": "{:.2f}".format(x * 105 - 52.5), 
                        "y": "{:.2f}".format(y * 68 - 34.0)
                    })
                except ValueError:
                    # loc の形式が不正な場合
                    raise ValueError(f"Invalid location format for player {player_id} in frame {frame_number}")

        return tracking_data

    def extract_tracking_data_from_json(json_path: str, period: str) -> List[Dict[str, Any]]:
        """
        Parse the JSON file and extract tracking data.

        Args:
            json_path (str): Path to the JSON file.
        Returns:
            list of dict: A list containing tracking information for each player in each frame.
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        tracking_data = []
        for frame_number, players in data.items():
            for player in players:
                try:
                    tracking_data.append({
                        "period": period,
                        "frame": int(frame_number),
                        "match_time": int(player.get("match_time", 0)),
                        "player_id": "ball" if player.get("player_id") == None else player.get("player_id"),
                        "x": "{:.2f}".format(float(player.get("x", 0) - 52.5)),
                        "y": "{:.2f}".format(float(player.get("y", 0) - 34.0))
                    })
                except ValueError:
                    raise ValueError(f"Invalid data format in frame {frame_number}")

        return tracking_data
    
    def devide_by_period(tracking_data_list: List[dict]) -> List[pd.DataFrame]:
        """
        トラッキングデータのリストに 'period' 列を追加し、periodごとに分割した
        DataFrameのリストを返す。
        
        frame番号が大きく減少する（リセットされる）ごとにperiodをインクリメントし、
        その直前の行で期間を終了する。

        Args:
            tracking_data_list (list of dict): tracking_dataを格納したリスト。
        Returns:
            List[pd.DataFrame]: 'period' 列が追加され、期間ごとに正確に分割されたDataFrameのリスト。
        """
        if not tracking_data_list:
            return []

        # 1. リストをPandas DataFrameに変換し、オリジナルのインデックスを保持
        df = pd.DataFrame(tracking_data_list)
        
        # 2. periodの境界となるインデックス（frame番号がリセットされる行）を特定
        # 各フレームの最初の行のみを取得
        first_occurrence_of_frame = df.drop_duplicates(subset=['frame', 'match_time'], keep='first')
        # frame番号の差分を計算し、負になる箇所（リセット）を検出
        # .diff() は Series を返すため、インデックスは first_occurrence_of_frame のインデックスと一致する
        frame_diff = first_occurrence_of_frame['frame'].diff().fillna(0)
        period_reset_indices = frame_diff[frame_diff < 0].index
        
        # 3. 分割点のインデックスリストを作成
        # リストの先頭 (0) を開始点として追加
        split_indices = [0]
        # リセットされたフレームのインデックスを取得
        # df.index.get_loc() を使わずに、直接 df のインデックスで操作する
        for reset_idx in period_reset_indices:
            # リセットが行われるフレームの直前のインデックスを分割点に追加
            # reset_idx は first_occurrence_of_frame のインデックスであり、df のインデックスと一致する
            if reset_idx > 0:
                split_indices.append(reset_idx) 
                
        # リストの末尾（データの最終インデックス+1）を終了点として追加
        split_indices.append(len(df))
        # 重複を削除し、ソート
        split_indices = sorted(list(set(split_indices)))
        period_df_list = []
        
        # 4. 分割とperiod番号の割り当て
        for i in range(len(split_indices) - 1):
            start_idx = split_indices[i]
            end_idx = split_indices[i+1]
            current_period = i + 1
            # DataFrameをスライス
            period_df = df.iloc[start_idx:end_idx].copy()
            # 'period' 列を割り当て
            period_df.loc[:, 'period'] = current_period
            # 不要な一時列をクリーンアップ（ここでは既に df に period がマッピングされていないので不要だが、念のため）
            period_df_list.append(period_df.reset_index(drop=True))

        return period_df_list
    
    def extract_meta_info_from_xml(xml_path: str) -> dict:
        """
        Extract team information (ID, name, side) from an XML metadata file.

        Args:
            xml_path (str): Path to the XML metadata file.
        Returns:
            dict: Dictionary in the format: {player_id: {'position': str, 'team_id': str, 'side': str}}.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        team_info = {}
        player_info = {}

        teams_element = root.find("teams")
        if teams_element is not None:
            for team in teams_element.findall("team"):
                team_id = team.get("id")
                team_name = team.get("name")
                side = team.get("side")
                
                if team_id:
                    team_info[team_id] = {
                        "team_name": team_name,
                        "side": side
                    }
        players_element = root.find("players")
        if players_element is not None:
            for player in players_element.findall("player"):
                player_id = player.get("id")
                player_name = player.get("name")
                team_id = player.get("teamId")
                position = player.get("position")
                
                if player_id:
                    side = team_info.get(team_id, {}).get("side")
                    team_name = team_info.get(team_id, {}).get("team_name")
                    
                    player_info[player_id] = {
                        "team_id": team_id,
                        "team_name": team_name,
                        "side": side,
                        "player_name": player_name,
                        "position": position,
                    }
        return player_info

    def extract_meta_info_from_json(json_path: str) -> dict:
        """
        Extract team information (ID, name, side) from an JSON metadata file.

        Args:
            xml_path (str): Path to the XML metadata file.
        Returns:
            dict: Dictionary in the format: {player_id: {'position': str, 'team_id': str, 'side': str}}.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        player_info = {}
        
        teams = {
            "home": data.get("home_team", {}),
            "away": data.get("away_team", {})
        }
        
        for side, team_data in teams.items():
            if team_data:
                team_id = str(team_data.get("team_id"))
                team_name = str(team_data.get("team_name"))
                
                # プレイヤー情報を保存
                if "players" in team_data:
                    for player in team_data["players"]:
                        player_id = str(player.get("player_id"))
                        player_name = str(player.get("full_name"))
                        position = player.get("initial_position_name")
                        
                        if player_id:
                            player_info[player_id] = {
                                "team_id": team_id,
                                "team_name": team_name,
                                "side": side,
                                "player_name": player_name,
                                "position": position,
                            }
                            
        return player_info

    def get_inplay_start_time(event_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 'inplay_num' column to event_df.
        If the first word in filtered_event_types matches the specified event type,
        it is considered the start of a new in-play event, and inplay_num is incremented.

        Args:
            event_df (pd.DataFrame): A DataFrame containing soccer event data.
        Returns:
            pd.DataFrame: A DataFrame with the 'inplay_num' column added.
        """
        
        event_df = event_df.copy()
        # List of strings specified as in-play start events
        START_EVENT_STRINGS = ['goalKick', 'throwIn', 'cornerKick', 'freeKick', 'goalAgainst']

        # 1. Extract the string up to the first space in 'filtered_event_types'
        # Since NaN values may be present, replace them with an empty string ('') before str.split().
        event_df.loc[:, 'first_event_type'] = event_df['filtered_event_types'].fillna('').str.split(' ').str[0]

        # 2. Create a flag column to detect the start frame
        # The first row is always considered the start of an in-play sequence
        is_start_frame = pd.Series(False, index=event_df.index)
        is_start_frame.iloc[0] = True
        # Detect events containing the specified strings
        is_restart_event = event_df['first_event_type'].isin(START_EVENT_STRINGS)

        # 3. Apply the restart logic
        # Restart events other than 'goalAgainst': The current row marks the start of a new in-play sequence
        is_normal_restart = is_restart_event & (event_df['first_event_type'] != 'goalAgainst')
        is_start_frame = is_start_frame | is_normal_restart
        # 'goalAgainst' restart event: The **next frame** marks the start of a new in-play sequence
        is_goal_against = event_df['first_event_type'] == 'goalAgainst'
        # Set True for the row immediately following 'goalAgainst' (using shift(-1), the last row is ignored)
        # This is OR combined with is_start_frame
        shifted_goal_against = is_goal_against.shift(periods=-1)
        filled_shifted = shifted_goal_against.fillna(False).astype(bool)
        is_start_frame = is_start_frame.astype(bool)
        is_start_frame = is_start_frame | filled_shifted

        # 4. Calculate the in-play number
        # Calculate the cumulative sum, which increments at every True (start frame) instance
        # Since True is treated as 1 and False as 0, cumsum() yields the in-play number
        event_df.loc[:, 'inplay_num'] = is_start_frame.cumsum().astype(int)

        # 5. Post-processing
        # Delete the helper column created during intermediate processing and return the result
        event_df = event_df.drop(columns=['first_event_type'], errors='ignore')

        return event_df
    
    def get_tracking(tracking_df: pd.DataFrame, event_df: pd.DataFrame, player_info_df: Dict[str, Dict[str, str]]) -> pd.DataFrame:
        """
        トラッキングデータをフレームごとに集約し、チームサイドとポジション順に並べ替えた
        ワイドフォーマットのDataFrameを作成し、インプレー番号を割り当てる。

        Args:
            tracking_df (pd.DataFrame): 処理されたトラッキングデータ (frame, period, x, y, player_idなどを含む)。
            event_df (pd.DataFrame): 処理されたイベントデータ (match_time, period, inplay_numなどを含む)。
            player_info_df (Dict[str, Dict[str, str]]): player_idに対するポジション、チームID、サイド情報を持つ辞書。

        Returns:
            pd.DataFrame: フレームごとのワイドフォーマットトラッキングデータ。
        """
        
        # 標準的なポジション順序 (1から11の番号付けに使用)
        POSITION_ORDER = ['GK', 'CB', 'RWB', 'RB', 'LWB', 'LB', 'CDM', 'RM', 'CM', 'LM', 'CAM', 'RW', 'LW', 'CF']
        FPS = 25 # トラッキングデータのフレームレート
        
        # -----------------------------------------------
        # 0. プレイヤー情報の結合と前処理
        # -----------------------------------------------
        event_df = event_df.copy()
        # player_info_dfをDataFrameに変換し、トラッキングデータにマージ
        player_map_df = pd.DataFrame.from_dict(player_info_df, orient='index').reset_index().rename(
            columns={'index': 'player_id', 'side': 'team_side', 'team_name': 'team_name'}
        )
        
        # player_idの型を揃える
        tracking_df['player_id'] = tracking_df['player_id'].astype(str)
        
        # プレイヤーのメタデータをトラッキングデータに結合
        tracking_df = pd.merge(tracking_df, player_map_df, on='player_id', how='left')

        # ボールの行のメタデータ ('player_id'='ball') を補完
        tracking_df.loc[tracking_df['player_id'] == 'ball', ['team_id', 'team_name', 'team_side', 'position', 'player_name']] = \
            ['ball', 'ball', 'ball', 'ball', 'ball']

        # -----------------------------------------------
        # 1. チームサイド (left/right) の決定 (最初のフレームで固定)
        # -----------------------------------------------
        
        # 最初のフレームのGKデータのみを抽出
        target_frame = tracking_df['frame'].min() + 10
        gk_data_initial = tracking_df[(tracking_df['position'] == 'GK') & (tracking_df['frame'] == target_frame)]
        
        # x座標が最小（マイナス側）のチームを 'left' チームとする
        left_team_id = gk_data_initial.loc[gk_data_initial['x'].idxmin(), 'team_id']
        
        # チームのメタデータを格納する辞書を作成（ワイドフォーマットの列作成に使用）
        team_meta = {}
        unique_teams = tracking_df[tracking_df['team_id'] != 'ball'][['team_id', 'team_name', 'team_side']].drop_duplicates()
        
        for _, row in unique_teams.iterrows():
            current_side = 'left' if row['team_id'] == left_team_id else 'right'
            
            team_meta[f'{current_side}_team_id'] = row['team_id']
            team_meta[f'{current_side}_team_name'] = row['team_name']
            team_meta[f'{current_side}_team_side'] = row['team_side'] # home/away

        # -----------------------------------------------
        # 2. インプレー番号 (inplay_num) の割り当てロジック
        # -----------------------------------------------

        # tracking_df に inplay_num 列を追加し、全て NaN で初期化
        # このコードは関数内での処理を想定しているため、DataFrameのコピーを直接修正します。
        tracking_df['inplay_num'] = np.nan

        # 1. event_dfから各インプレーの開始/終了時刻を決定

        # 'inplay_num' と 'match_time' の組み合わせを取得し、インプレー番号でソート
        inplay_times = event_df[['inplay_num', 'event_time']].drop_duplicates().sort_values('inplay_num')
        # 各インプレー番号の開始時刻と終了時刻を計算
        inplay_periods = inplay_times.groupby('inplay_num')['event_time'].agg(['min', 'max']).reset_index()
        inplay_periods.columns = ['inplay_num', 'start_time', 'end_time']

        # 2. tracking_df に inplay_num を割り当て

        # Period ごとに処理を行い、割り当てを確実にする
        for period in tracking_df['period'].unique():
            
            # 当該ピリオドの tracking_df を抽出
            p_tracking = tracking_df[tracking_df['period'] == period].copy()
            
            # 当該ピリオドのインプレー期間を抽出
            p_inplay_periods = inplay_periods.copy()
            
            # 各インプレー期間に対して tracking_df に inplay_num を割り当て
            for _, row in p_inplay_periods.iterrows():
                current_inplay_num = row['inplay_num']
                start_time = row['start_time']
                end_time = row['end_time']
                
                # 'match_time' が 'start_time' 以上かつ 'true_end_time' 以下のフレームに 'inplay_num' を設定
                # NumPyのwhere条件を使用して高速に処理
                
                # グローバルな tracking_df のインデックスを取得
                mask_index = tracking_df[
                    (tracking_df['period'] == period) & 
                    (tracking_df['match_time'] >= start_time) & 
                    (tracking_df['match_time'] <= end_time)
                ].index
                
                # マスクされた行に inplay_num を割り当てる
                tracking_df.loc[mask_index, 'inplay_num'] = current_inplay_num

        # 割り当てられなかった NaN の inplay_num はインプレー間の中断フレームと見なされます。
        # 最終的な final_tracking_df は tracking_df そのものです。
        final_tracking_df = tracking_df.copy()

        # -----------------------------------------------
        # 3. プレイヤー順序の決定と結合キーの作成
        # -----------------------------------------------
        
        is_player = (final_tracking_df['player_id'] != 'ball')
        side_calculated = np.where(
            final_tracking_df['team_id'] == left_team_id,
            'left',
            'right'
        )
        side_series = pd.Series(side_calculated, index=final_tracking_df.index)
        if 'side' not in final_tracking_df.columns:
            final_tracking_df.loc[:, 'side'] = np.nan
        final_tracking_df['side'] = final_tracking_df['side'].astype(object)
        final_tracking_df.loc[is_player, 'side'] = side_series.loc[is_player]
        final_tracking_df.loc[final_tracking_df['player_id'] == 'ball', 'side'] = 'ball'

        # ポジションの順序をマッピング
        pos_map = {pos: order for order, pos in enumerate(POSITION_ORDER, 1)}
        
        # プレイヤーのみをフィルタリング
        player_df = final_tracking_df[final_tracking_df['player_id'] != 'ball'].copy()
        
        # ポジションの順序番号をDataFrameに追加
        player_df.loc[:, 'pos_order'] = player_df['position'].map(pos_map)
        
        # 各チーム・各フレーム内でポジション順に連番 (1から11) を作成
        player_df.loc[:, 'pos_rank'] = player_df.groupby(['frame', 'side'])['pos_order'].rank(method='first').astype(int)
        
        # ワイドフォーマットの列名を作成: 例: 'left_1_x', 'right_11_y'
        player_df.loc[:, 'variable'] = player_df['side'] + '_' + player_df['pos_rank'].astype(str)
        
        # -----------------------------------------------
        # 4. プレイヤーデータのワイドフォーマット化 (Pivot)
        # -----------------------------------------------
        
        # ワイド化する値列をリスト化
        value_cols = ['x', 'y', 'player_id', 'player_name', 'position']
        
        wide_data_list = []
        
        for col in value_cols:
            pivot_df = player_df.pivot_table(
                index=['frame', 'match_time', 'period', 'inplay_num'], 
                columns='variable', 
                values=col,
                aggfunc='first'
            ).add_suffix(f'_{col.replace("player_id", "id").replace("player_name", "name")}') # player_id -> left_1_id

            wide_data_list.append(pivot_df)

        # 全てのピボットテーブルを結合
        wide_player_df = wide_data_list[0].join(wide_data_list[1:])
        
        # -----------------------------------------------
        # 5. ボールデータとチームメタデータの抽出・結合
        # -----------------------------------------------
        
        # ボールデータを抽出
        ball_df = final_tracking_df[final_tracking_df['player_id'] == 'ball'][['frame', 'x', 'y', 'match_time', 'period', 'inplay_num']].rename(
            columns={'x': 'ball_x', 'y': 'ball_y'}
        ).set_index(['frame', 'match_time', 'period', 'inplay_num'])
        
        # プレイヤーデータにボールデータを結合
        final_tracking_df = wide_player_df.join(ball_df).reset_index()
        
        # チームメタデータを追加
        for col, value in team_meta.items():
            final_tracking_df[col] = value

        # -----------------------------------------------
        # 6. 最終的な列の整形と順序調整
        # -----------------------------------------------

        # プレイヤー列を ID, Name, Position, x, y の順で生成
        ordered_player_cols = []
        for side in ['left', 'right']:
            for i in range(1, 12): # 1番から11番まで
                prefix = f'{side}_{i}_'
                
                # ID, Name, Positionはデータに存在しない可能性もあるため、チェックしてから追加
                ordered_player_cols.append(prefix + 'id')
                ordered_player_cols.append(prefix + 'name')
                ordered_player_cols.append(prefix + 'position')
                ordered_player_cols.append(prefix + 'x')
                ordered_player_cols.append(prefix + 'y')

        # 最終的な列順序 (要望の形式に合わせる)
        base_cols = ['period', 'inplay_num', 'frame', 'match_time', 'ball_x', 'ball_y']
        
        # チームメタデータ列
        team_cols = []
        for side in ['left', 'right']:
            team_cols.extend([f'{side}_team_id', f'{side}_team_name', f'{side}_team_side'])
            
        final_cols = base_cols + team_cols + ordered_player_cols
        
        # 必要な列のみを選択し、順序を調整 (存在しない列は無視される)
        final_tracking_df = final_tracking_df.reindex(columns=final_cols)
        
        return final_tracking_df
    
    # Load the event data
    event_df = pd.read_csv(event_path)
    # devide by period
    grouped_events = event_df.groupby('event_period')
    PERIOD_ORDER = ['FIRST_HALF', 'SECOND_HALF', 'EXTRA_FIRST_HALF', 'EXTRA_SECOND_HALF']
    # check if the format is the latest version
    if tracking_xml_path is None:
        list_of_tracking_data = []
        for i in range(len(tracking_json_paths)):
            tracking_data = extract_tracking_data_from_json(tracking_json_paths[i], period=str(i+1))
            list_of_tracking_data.append(tracking_data)
        player_info_df = extract_meta_info_from_json(meta_data_path)
    else:
        tracking_data = extract_tracking_data_from_xml(tracking_xml_path)
        # add period
        list_of_tracking_data = devide_by_period(tracking_data)
        player_info_df = extract_meta_info_from_xml(meta_data_path)
    
    final_tracking_df_list = []
    for i in range(len(list_of_tracking_data)):
        event_df = grouped_events.get_group(PERIOD_ORDER[i])
        tracking_df = pd.DataFrame(list_of_tracking_data[i])
        # Get additional features
        event_df = get_inplay_start_time(event_df)
        # Get tracking features
        processed_tracking_df = get_tracking(tracking_df, event_df, player_info_df)
        final_tracking_df_list.append(processed_tracking_df)
    
    final_tracking_df = pd.concat(final_tracking_df_list, ignore_index=True)
    return final_tracking_df

def load_statsbomb_skillcorner(sb_event_path: str, sc_tracking_path: str, sc_match_path: str, sc_players_path: str) -> pd.DataFrame:
    """
    Load and merge StatsBomb event data with SkillCorner tracking data.

    Args:
        statsbomb_event_dir (str): Directory path for StatsBomb event data.
        skillcorner_tracking_dir (str): Directory path for SkillCorner tracking data.
        skillcorner_match_dir (str): Directory path for SkillCorner match data.
        statsbomb_match_id (str): Match ID for StatsBomb data.
        skillcorner_match_id (str): Match ID for SkillCorner data.

    Returns:
        pd.DataFrame: Combined DataFrame with event and tracking data.
    """
    
    def extract_meta_info_from_match(sc_match: dict, sc_players: list) -> dict:
        """
        Extract team and player information (ID, name, side) from a json match data file.

        Args:
            sc_match (dict): Dataframe of match data file.
        Returns:
            dict: Dictionary in the format: {team_id: {'team_name': str, 'team_side': str}}, {player_id: {'position': str, 'team_id': str, 'side': str}}.
        """
        # 結果を格納する辞書の初期化
        team_meta_df = {}
        player_meta_df = {}

        player_trackable_map = {p['id']: p.get('trackable_object') for p in sc_players}

        # 1. チーム情報の作成
        # Home Team
        home_id = sc_match['home_team']['id']
        team_meta_df[home_id] = {
            'team_name': sc_match['home_team']['name'],
            'team_side': 'home'
        }

        # Away Team
        away_id = sc_match['away_team']['id']
        team_meta_df[away_id] = {
            'team_name': sc_match['away_team']['name'],
            'team_side': 'away'
        }

        # 2. 選手情報の作成
        for p in sc_match['players']:
            player_id = p['id']
            trackable_id = player_trackable_map.get(player_id)
            player_meta_df[trackable_id] = {
                'team_id': p['team_id'],
                'player_name': p['short_name'],
                'position_name': p['player_role']['name'],
                'position_acronym': p['player_role']['acronym']
            }

        return team_meta_df, player_meta_df

    def get_left_team_id(sc_tracking, team_meta_df, player_meta_df):
        all_team_ids = list(team_meta_df.keys())
        for frame_data in sc_tracking:
            if frame_data['data']==None:
                continue
            for obj in frame_data['data']:
                if 'z' in obj:
                    continue
                p_id = obj['trackable_object']
                p_info = player_meta_df[p_id]
                if p_info['position_acronym'] == 'GK':
                    if obj['x'] < 0.0:
                        left_team_id = p_info['team_id']
                    else:
                        left_team_id = [tid for tid in all_team_ids if tid != p_info['team_id']][0] 
                    return left_team_id
        return None

    def process_all_tracking(sc_tracking, team_meta_df, player_meta_df, left_team_id):
        """
        全フレームをループし、ポジション順にソートされたフラットなリストを返す。
        """

        # ポジションの優先順位を辞書化（スコアが低いほど若い番号に割り当てられる）
        POSITION_ORDER = ['GK', 'CB', 'RCB', 'LCB', 'RWB', 'RB', 'LWB', 'LB', 'CDM', 'RDM', 'LDM', 'RM', 'CM', 'LM', 'CAM', 'RW', 'LW', 'CF']
        pos_priority = {pos: i for i, pos in enumerate(POSITION_ORDER)}

        # 左右のチームIDを特定
        all_team_ids = list(team_meta_df.keys())
        right_team_id = [tid for tid in all_team_ids if tid != left_team_id][0]
        
        all_frames_processed = []

        for frame_data in sc_tracking:
            # 基本情報の構築
            res = {
                'period': int(frame_data['period']) if pd.notna(frame_data['period']) else None,
                'inplay_num': None, # 予約列
                'frame': frame_data['frame'],
                'match_time': frame_data['timestamp'],
                'ball_x': None, # 後で更新
                'ball_y': None, # 後で更新
                'left_team_id': left_team_id,
                'left_team_name': team_meta_df[left_team_id]['team_name'],
                'left_team_side': team_meta_df[left_team_id]['team_side'],
                'right_team_id': right_team_id,
                'right_team_name': team_meta_df[right_team_id]['team_name'],
                'right_team_side': team_meta_df[right_team_id]['team_side']
            }

            # フレーム内のデータを「ボール」と「左右の選手リスト」に分ける
            left_players_in_frame = []
            right_players_in_frame = []
            
            for obj in frame_data['data']:
                # ボールの処理
                if 'z' in obj:
                    res['ball_x'] = obj['x']
                    res['ball_y'] = obj['y']
                    continue
                
                # 選手の処理
                p_id = obj['track_id']
                if p_id in player_meta_df:
                    p_info = player_meta_df[p_id]
                    player_data = {
                        'id': p_id,
                        'name': p_info['player_name'],
                        'pos': p_info['position_acronym'],
                        'x': obj['x'],
                        'y': obj['y'],
                        'priority': pos_priority.get(p_info['position_acronym'], 99) # 未定義は最後尾
                    }
                    
                    if p_info['team_id'] == left_team_id:
                        left_players_in_frame.append(player_data)
                    else:
                        right_players_in_frame.append(player_data)

            # -------------------------------------------------------
            # ⭐ ポジション順（同ポジションならID順）でソート
            # -------------------------------------------------------
            left_players_sorted = sorted(left_players_in_frame, key=lambda x: (x['priority'], x['id']))
            right_players_sorted = sorted(right_players_in_frame, key=lambda x: (x['priority'], x['id']))

            # ソートされた順に left_1, left_2 ... と格納 (最大11人)
            for i in range(11):
                idx = i + 1
                # Left Team
                if i < len(left_players_sorted):
                    p = left_players_sorted[i]
                    res[f"left_{idx}_id"] = p['id']
                    res[f"left_{idx}_name"] = p['name']
                    res[f"left_{idx}_position"] = p['pos']
                    res[f"left_{idx}_x"] = p['x']
                    res[f"left_{idx}_y"] = p['y']
                else:
                    # 11人に満たない場合はNaNで埋める（列順を維持するため重要）
                    res[f"left_{idx}_id"] = None
                    res[f"left_{idx}_name"] = None
                    res[f"left_{idx}_position"] = None
                    res[f"left_{idx}_x"] = None
                    res[f"left_{idx}_y"] = None

            for i in range(11):
                idx = i + 1
                # Right Team
                if i < len(right_players_sorted):
                    p = right_players_sorted[i]
                    res[f"right_{idx}_id"] = p['id']
                    res[f"right_{idx}_name"] = p['name']
                    res[f"right_{idx}_position"] = p['pos']
                    res[f"right_{idx}_x"] = p['x']
                    res[f"right_{idx}_y"] = p['y']
                else:
                    res[f"right_{idx}_id"] = None
                    res[f"right_{idx}_name"] = None
                    res[f"right_{idx}_position"] = None
                    res[f"right_{idx}_x"] = None
                    res[f"right_{idx}_y"] = None

            all_frames_processed.append(res)

        return pd.DataFrame(all_frames_processed)
    
    def get_inplay_start_time(event_df: pd.DataFrame):
        """
        event_dfにinplay_numを追加し、各インプレーの開始情報を辞書のリストで返す。
        """
        # データのコピーを作成
        df = event_df.copy()
        
        # 開始情報を保持するリスト（辞書を格納）
        inplay_info_list = []
        
        # インプレー番号を初期化
        current_inplay = 0

        continuing_patterns = ['Regular Play', 'From Counter', 'From Keeper']
        restart_types = ['Throw-in', 'Corner', 'Goal Kick', 'Free Kick']

        for i in range(len(df) - 1):
            curr_ev = df.iloc[i]
            next_ev = df.iloc[i + 1]

            # pass_type が None の場合は判定をスキップ（元のロジックを維持）
            if pd.isna(next_ev['pass_type']):
                continue
            
            # --- インプレーの切り替わり条件判定 ---
            is_new_inplay = False

            # 1. 試合終了後のデータ（時間が戻る場合）対策
            next_ts = pd.Timestamp(next_ev['timestamp']).round('100ms')
            curr_ts = pd.Timestamp(curr_ev['timestamp']).round('100ms')
            if next_ts < curr_ts:
                is_new_inplay = True

            # 条件A: play_patternの変化
            elif curr_ev['play_pattern'] != next_ev['play_pattern']:
                if next_ev['play_pattern'] not in continuing_patterns:
                    is_new_inplay = True

            # 条件B: 特定の再開イベント
            elif next_ev['pass_type'] in restart_types:
                is_new_inplay = True

            # --- インプレー番号の更新と情報の記録 ---
            if is_new_inplay:
                current_inplay += 1
                # 必要な情報を辞書形式で保存
                inplay_info_list.append({
                    'inplay_num': current_inplay,
                    'period': int(next_ev['period']),
                    'timestamp': next_ts
                })

        return inplay_info_list
    
    def get_inplay_tracking(tracking_df: pd.DataFrame, inplay_info_list: List) -> pd.DataFrame:
        """
        inplay_info_listを元に、トラッキングデータにinplay_numを付与し、
        インプレー外（区間外）のデータを削除する。
        """
        df = tracking_df.copy()

        # 1. トラッキングデータの時間を統一された日付（1900-01-01）のTimestampに変換
        # これにより「時間・分・秒」のみの純粋な比較が可能になります
        df['tmp_timestamp'] = pd.to_datetime(
            df['match_time'], format='%H:%M:%S.%f', errors='coerce'
        ).map(lambda x: x.replace(year=1900, month=1, day=1) if pd.notna(x) else x)

        # 2. インプレー情報の時間も同じ日付（1900-01-01）に統一
        def normalize_period_time(group):
            period_start = group['tmp_timestamp'].min()
            # 経過時間を計算し、1900-01-01 00:00:00 からの経過に変換し直す
            base = pd.Timestamp('1900-01-01 00:00:00')
            group['tmp_timestamp'] = base + (group['tmp_timestamp'] - period_start)
            return group

        def normalize_time(ts):
            if isinstance(ts, pd.Timestamp):
                return ts.replace(year=1900, month=1, day=1)
            return ts
        
        df = df.groupby('period', group_keys=False).apply(normalize_period_time)

        # --- インプレー番号の割り当て ---
        for i in range(len(inplay_info_list)):
            current_info = inplay_info_list[i]
            
            # 日付を1900-01-01に揃える
            start_time = normalize_time(current_info['timestamp'])
            period = current_info['period']
            num = current_info['inplay_num']

            period_mask = (df['period'] == period)

            # 次のインプレー開始時間を取得
            next_event_in_same_period = None
            for j in range(i + 1, len(inplay_info_list)):
                if int(inplay_info_list[j]['period']) == period:
                    next_event_in_same_period = normalize_time(inplay_info_list[j]['timestamp'])
                    break
            
            if next_event_in_same_period is not None:
                # 同じピリオド内に次のインプレーがある場合: その直前まで
                time_mask = (df['tmp_timestamp'] >= start_time) & (df['tmp_timestamp'] < next_event_in_same_period)
            else:
                # そのピリオド内で最後のインプレーの場合: ピリオドの最後まで
                time_mask = (df['tmp_timestamp'] >= start_time)

            final_mask = period_mask & time_mask
            df.loc[final_mask, 'inplay_num'] = num

        # --- データのクリーンアップ ---
        # inplay_num が割り当てられなかった行（インプレー外）を削除
        df = df.dropna(subset=['inplay_num'])

        # tmp_timestamp を文字列フォーマットに戻す (%f はマイクロ秒なので下3桁をカット)
        base_time = pd.Timestamp('1900-01-01 00:00:00')
        df['match_time'] = (df['tmp_timestamp'] - base_time).dt.total_seconds() * 1000
        df = df[df['match_time'] % 200 == 0]
        df['match_time'] = df['match_time'].astype(int)
        df = df.drop(columns=['tmp_timestamp'])

        # 型を整数に戻す
        df['period'] = df['period'].astype(int)
        df['inplay_num'] = df['inplay_num'].astype(int)

        return df.reset_index(drop=True)
    
    # Load the event data
    with open(sb_event_path, 'rb') as f:
        sb_event = pickle.load(f)
    with open(sc_tracking_path, 'r', encoding='utf-8') as f:
        sc_tracking = json.load(f)
    with open(sc_match_path, 'r', encoding='utf-8') as f:
        sc_match = json.load(f)
    with open(sc_players_path, 'r', encoding='utf-8') as f:
        sc_players = json.load(f)

    team_meta_df, player_meta_df = extract_meta_info_from_match(sc_match, sc_players)

    left_team_id = get_left_team_id(sc_tracking, team_meta_df, player_meta_df)

    tracking_df = process_all_tracking(sc_tracking, team_meta_df, player_meta_df, left_team_id)

    inplay_info_list = get_inplay_start_time(sb_event)

    processed_tracking_df = get_inplay_tracking(tracking_df, inplay_info_list)

    return processed_tracking_df

def load_pff2metrica(event_path:str, match_id:str = None) -> pd.DataFrame:
    """
    Convert PFF-style event data to Metrica format.

    Parameters
    ----------
    event_df : pd.DataFrame
        Event data from PFF dataset with columns like:
        - gameEvents_period
        - gameEvents_playerName
        - possessionEvents_receiverPlayerName
        - possessionEvents_possessionEventType
        - startTime, endTime, duration
        - gameEvents_homeTeam
        - various outcome types for success/failure
    match_id : str, optional
        Match identifier to add as a column, by default None

    Returns
    -------
    Metrica_df : pd.DataFrame
        DataFrame in Metrica format with columns:
        ['Team', 'Type', 'Subtype', 'Period', 'Start Frame', 'Start Time [s]',
         'End Frame', 'End Time [s]', 'From', 'To', 'Start X', 'Start Y', 'End X', 'End Y']
    """
    with open(event_path, 'r') as f:
        event_data = json.load(f)
        event_df = pd.json_normalize(event_data, sep='_')
    
    def type_id2name(x):
        """
        Map event type codes to descriptive names.

        Parameters
        ----------
        x : str | int | float | None
            Event type code (e.g., 'PA', 'SH', 'FO', etc.)

        Returns
        -------
        str | None
            Descriptive event type name, or None if not mapped.
        """
        import math
        if x in ['PA']:
            x = "pass"
        elif x in ['CR']:
            x = "cross"
        # elif x == 2:
        #     x = "throw_in"
        # elif x == 5:
        #     x = "corner_crossed"
        # elif x == 7:
        #     x = "take_on"
        elif x in ['FO']:
            x = "foul"
        elif x in ['CH']:
            x = "tackle"
        # elif x == 10:
        #     x = "interception"
        elif x in ['SH']:
            x = "shot"
        elif x in ['CL']:
            x = "clearance"
        elif x in ['BC']:
            x = "dribble"
        # elif x == 22:
        #     x = "goalkick"
        elif x in ['IT', 'RE', 'TC']:
            x = "other"
        elif x is None or (isinstance(x, (float, int)) and math.isnan(x)):
            x = None
        else:
            print(f"Unmapped event type: {x}")
        return x
    def extract_player_xy(row):
        """
        Extracts the (x, y) coordinates of the player involved in a game event.

        Parameters
        ----------
        row : pd.Series
            A row from a DataFrame containing game event and player information. 
            Expected keys:
                - "gameEvents_homeTeam" (bool): True if home team, False if away team.
                - "homePlayers" (list|str): List or stringified list of home team players.
                - "awayPlayers" (list|str): List or stringified list of away team players.
                - "gameEvents_playerId" (int): ID of the player involved in the event.

        Returns
        -------
        pd.Series
            A Series with coordinates:
            - "start_x"
            - "start_y"
            - "end_x"
            - "end_y"
            If the player is not found, all values are None.
        """
        # choose player list
        if row["gameEvents_homeTeam"] is True:
            player_dict = row["homePlayers"]
        elif row["gameEvents_homeTeam"] is False:
            player_dict = row["awayPlayers"]
        else:
            return pd.Series([None, None, None, None], index=["start_x", "start_y", "end_x", "end_y"])
        
        # find target player
        player_dict = ast.literal_eval(player_dict) if type(player_dict) == str else player_dict
        target_player = next((d for d in player_dict if d["playerId"] == row["gameEvents_playerId"]), None)

        if target_player:
            return pd.Series(
                [target_player["x"], target_player["y"], target_player["x"], target_player["y"]],
                index=["start_x", "start_y", "end_x", "end_y"]
            )
        else:
            return pd.Series([None, None, None, None], index=["start_x", "start_y", "end_x", "end_y"])

    # drop row where gameEvents_startGameClock is NaN
    event_df = event_df.dropna(subset=['gameEvents_startGameClock']).reset_index(drop=True)

    # set column name
    column_name = ['Team', 
          'Type',
          'Subtype',
          'Period',
          'Start Frame',
          'Start Time [s]',
          'End Frame',
          'End Time [s]',
          'From',
          'To',
          'Start X',
          'Start Y',
          'End X',
          'End Y']
    Metrica_df = pd.DataFrame(columns=column_name)
    Metrica_df['Period'] = event_df['gameEvents_period']
    event_df[["start_x", "start_y", "end_x", "end_y"]] = event_df.apply(extract_player_xy, axis=1)
    Metrica_df['Start X'] = event_df['start_x'] #- 52.5
    Metrica_df['Start Y'] = event_df['start_y'] #- 34
    Metrica_df['End X'] = event_df['end_x'] #- 52.5
    Metrica_df['End Y'] = event_df['end_y'] #- 34
    Metrica_df['From'] = event_df['gameEvents_playerName']
    Metrica_df['To'] = event_df['possessionEvents_receiverPlayerName']
    Metrica_df['Type'] = event_df['possessionEvents_possessionEventType']
    Metrica_df['Type'] = Metrica_df['Type'].apply(type_id2name)

    idx = event_df.index

    def col(name):
        """Safe getter: returns Series aligned to df (all NaN if col missing)."""
        return event_df[name] if name in event_df.columns else pd.Series(pd.NA, index=idx)

    # Raw outcome columns
    pass_out   = col('possessionEvents_passOutcomeType')       
    cross_out  = col('possessionEvents_crossOutcomeType')       
    shot_out   = col('possessionEvents_shotOutcomeType')        
    clr_out    = col('possessionEvents_clearanceOutcomeType')  
    tkl_out    = col('possessionEvents_challengeOutcomeType')   
    carry_out  = col('possessionEvents_ballCarryOutcome')       
    touch_out  = col('possessionEvents_touchOutcomeType')       

    # Per-action success masks (nullable booleans)
    event_df['pass_success']      = pass_out.isin(['C'])
    event_df['cross_success']     = cross_out.isin(['C'])
    event_df['shot_success']      = shot_out.isin(['G'])
    event_df['clearance_success'] = ~clr_out.isin(['B','D']) & clr_out.notna()
    event_df['tackle_success']    = tkl_out.isin(['B','C','M'])
    event_df['dribble_success']   = carry_out.isin(['R'])
    event_df['touch_success']     = touch_out.isin(['R'])

    # Where each action is *present* (not NaN), assign Subtype based on its success
    event_df['Subtype'] = np.nan

    def apply_subtype(success_col, present_series):
        """Set Subtype for rows where this action is present."""
        is_present = present_series.notna()
        success    = event_df[success_col] == True
        fail       = event_df[success_col] == False
        event_df.loc[is_present & success, 'Subtype'] = 'success'
        event_df.loc[is_present & fail,    'Subtype'] = 'fail'

    apply_subtype('pass_success',      pass_out)
    apply_subtype('cross_success',     cross_out)
    apply_subtype('shot_success',      shot_out)
    apply_subtype('clearance_success', clr_out)
    apply_subtype('tackle_success',    tkl_out)
    apply_subtype('dribble_success',   carry_out)
    apply_subtype('touch_success',     touch_out)
    Metrica_df['Subtype'] = event_df['Subtype']

    fps = 29.97

    Metrica_df['Start Time [s]'] = (event_df['gameEvents_startGameClock']).round().astype(int)
    Metrica_df['End Time [s]'] = (event_df['duration'] + event_df['gameEvents_startGameClock']).round().astype(int)

    Metrica_df['Start Frame'] = ((event_df['startTime'] - event_df['startTime'][0]) * fps).round().astype(int)
    end_frame = ((event_df['endTime'] - event_df['startTime'][0]) * fps).round()
    Metrica_df['End Frame'] = end_frame.fillna(Metrica_df['Start Frame']).astype(int)
    Metrica_df['Team'] = np.where(event_df['gameEvents_homeTeam'] == True, 'Home',
                      np.where(event_df['gameEvents_homeTeam'] == False, 'Away', None))

    #drop rows where start_x or start_y is NaN
    Metrica_df = Metrica_df.dropna(subset=['Start X', 'Start Y'])
    Metrica_df = Metrica_df.reset_index(drop=True)

    if match_id is not None:
        Metrica_df['match_id'] = match_id
        cols = Metrica_df.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        Metrica_df = Metrica_df[cols]

    return Metrica_df
    """
    Load event data from CSV file and optionally merge with tracking data.

    Args:
        event_path (str): Path to the CSV file containing event data.
        match_id (str, optional): Identifier for the match. Defaults to None.
        tracking_path (str, optional): Path to the CSV file containing tracking data. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing event and tracking data.
    """
    # Load event data from CSV file
    event_df = pd.read_csv(event_path)
    
    # Load tracking data if provided
    if tracking_path:
        tracking_df = pd.read_csv(tracking_path)
    
    # Define columns for the DataFrame
    columns = ["match_id", "seconds", "event_type", "outcome", "team", "player", "start_x", "start_y", "end_x", "end_y"]
    if tracking_path:
        columns.extend([" l_score", " r_score", " b_x", " b_y"])
        for i in range(1, 12):
            columns.extend([f" l{i}_x", f" l{i}_y"])
        for i in range(1, 12):
            columns.extend([f" r{i}_x", f" r{i}_y"])
        
    
    # Initialize an empty list to store event details
    event_list = []
    
    # Iterate through event records
    for index, record in event_df.iterrows():
        seconds = record.get('Time1', None)
        event_type = record.get('Type', None)
        outcome = record.get('Success', None)
        team = record.get('Side1', None)
        player = record.get('Unum1', None)
        start_x = record.get('X1', None)
        start_y = record.get('Y1', None)
        end_x = record.get('X2', None)
        end_y = record.get('Y2', None)
        
        # If tracking data is provided, merge with event details
        if tracking_path:
            if seconds in tracking_df[' cycle'].values:
                tracking_record = tracking_df[tracking_df[' cycle'] == seconds]
                if tracking_record.shape[0] != 1:
                    print(f"Error: Tracking record {index} has more than one row")
                    continue
                
                # Extract tracking data
                tracking_values = tracking_record.iloc[0].to_dict()

                # tracking_values.pop(' cycle')  # Remove the cycle column
                tracking_values = {key: value for key, value in tracking_values.items() if key in columns}
                # Append event and tracking details to the list
                event_list.append([match_id, seconds, event_type, outcome, team, player, start_x, start_y, end_x, end_y, *tracking_values.values()])
        else:
            # Append only event details
            event_list.append([match_id, seconds, event_type, outcome, team, player, start_x, start_y, end_x, end_y])
    
    # Convert the event list to a DataFrame
    df = pd.DataFrame(event_list, columns=columns)
    
    # Sort the DataFrame by 'seconds'
    df = df.sort_values(by="seconds").reset_index(drop=True)
    
    return df