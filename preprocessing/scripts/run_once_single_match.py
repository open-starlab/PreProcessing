from pathlib import Path
from config import PipelineConfig
from preprocessing import preprocess_match
from feature_engineering import compute_match_features
from main import (
    save_preprocessed_data,
    save_features,
    generate_summary_statistics,
    save_summary_statistics,
    save_transition_processing_log,
)

config = PipelineConfig(
    data_match='all_matches',
    back_four='all_players',
    sequence_type='negative_transition',
    reward_features='4_features',
    method='feature_computation',
    data_dir='/home/s_dash/workspace6/Defense_line/Laliga2023/24',
    output_dir='/home/s_dash/workspace6/processed_features'
)

match_dirs = sorted([d for d in Path(config.data_dir).iterdir() if d.is_dir() and d.name.isdigit()])
if not match_dirs:
    raise RuntimeError('No match directories found')

match_dir = match_dirs[0]
print(f'RUN_MATCH={match_dir.name}')
match_data = preprocess_match(match_dir, config)
if match_data is None:
    raise RuntimeError('preprocess_match returned None')

features_df = compute_match_features(match_data, config)
if features_df.empty:
    raise RuntimeError('No features computed for selected match')

features_df['match_id'] = match_data['match_id']
features_df['home_team'] = match_data['home_team']
features_df['away_team'] = match_data['away_team']

pre_path = save_preprocessed_data([match_data], Path(config.output_dir), config)
feat_path = save_features(features_df, Path(config.output_dir), config)
overall_df, team_df = generate_summary_statistics(features_df)
summary_paths = save_summary_statistics(overall_df, team_df, Path(config.output_dir), config)
log_paths = save_transition_processing_log([match_data], features_df, Path(config.output_dir), config)

preview_cols = [
    'match_id', 'transition_idx', 'label', 'home_team', 'away_team',
    'team_lost_possession', 'team_gained_possession', 'period', 'player_lost_possession'
]
preview_cols += [c for c in features_df.columns if c.endswith('_mean')]
preview_cols = [c for c in preview_cols if c in features_df.columns]

print('FEATURE_ROWS=', len(features_df))
print('FEATURE_COLS=', len(features_df.columns))
print('PREPROCESSED_PATH=', pre_path)
print('FEATURES_PATH=', feat_path)
print('SUMMARY_OVERALL_PATH=', summary_paths['overall'])
print('SUMMARY_TEAM_PATH=', summary_paths['team'])
print('LOG_CSV_PATH=', log_paths['csv'])
print('LOG_JSON_PATH=', log_paths['json'])
print('PREVIEW_HEAD_START')
print(features_df[preview_cols].head(5).to_string(index=False))
print('PREVIEW_HEAD_END')
