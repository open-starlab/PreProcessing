from pathlib import Path
import json
import traceback

from config import PipelineConfig
from preprocessing import preprocess_match
from feature_engineering import compute_match_features

result = {
    "ok": False,
    "error": None,
    "preprocess_ok": False,
    "is_sar": False,
    "states_shape": None,
    "actions_shape": None,
    "rewards_shape": None,
    "metadata_has_sequence_metadata": False,
    "sequence_metadata_count": 0,
    "required_present": False,
    "sample_keys": []
}

try:
    match_dir = Path('/home/s_dash/workspace6/Defense_line/Laliga2023/24/1018887')
    config = PipelineConfig(
        data_match='all_matches',
        back_four='all_players',
        sequence_type='negative_transition',
        reward_features='4_features',
        method='girl',
        data_dir='/home/s_dash/workspace6/Defense_line/Laliga2023/24',
        output_dir='/home/s_dash/workspace6/cleaned/output',
    )

    m = preprocess_match(match_dir, config)
    result["preprocess_ok"] = m is not None
    if m is None:
        raise RuntimeError("preprocess_match returned None")

    out = compute_match_features(m, config)
    result["is_sar"] = isinstance(out, dict) and bool(out.get('is_sar'))
    if not (isinstance(out, dict) and out.get('is_sar') and out.get('sar_data') is not None):
        raise RuntimeError("compute_match_features did not return SAR dict with sar_data")

    sd = out['sar_data']
    result["states_shape"] = tuple(sd['states'].shape)
    result["actions_shape"] = tuple(sd['actions'].shape)
    result["rewards_shape"] = tuple(sd['rewards'].shape)

    meta = sd['metadata']
    result["metadata_has_sequence_metadata"] = 'sequence_metadata' in meta
    seq_meta = meta.get('sequence_metadata', [])
    result["sequence_metadata_count"] = len(seq_meta)

    if seq_meta:
        sample = seq_meta[0]
        req = ['match_id','team_id','sequence_id','home_team','away_team','label','start_frame','end_frame']
        result["required_present"] = all(k in sample for k in req)
        result["sample_keys"] = sorted(sample.keys())

    result["ok"] = (
        result["is_sar"]
        and isinstance(result["states_shape"], tuple)
        and len(result["states_shape"]) == 3
        and result["states_shape"][1] == 10
        and result["states_shape"][2] == 21
        and isinstance(result["actions_shape"], tuple)
        and len(result["actions_shape"]) == 2
        and result["actions_shape"][1] == 10
        and isinstance(result["rewards_shape"], tuple)
        and len(result["rewards_shape"]) == 3
        and result["rewards_shape"][1] == 10
        and result["rewards_shape"][2] == 4
        and result["metadata_has_sequence_metadata"]
        and result["required_present"]
    )
except Exception as e:
    result["error"] = str(e)
    result["traceback"] = traceback.format_exc()

out_path = Path('/home/s_dash/workspace6/cleaned/output/check_sar_result.json')
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(result, indent=2))
print(f"WROTE={out_path}")
print(json.dumps(result, indent=2))
