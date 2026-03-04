#!/usr/bin/env python
from feature_pipeline.utils.data_loader import load_feature_dataset
from feature_pipeline.stats.two_way_anova import run_two_way_anova
from pathlib import Path

X, y, team, df = load_feature_dataset()
anova_df, corr_df = run_two_way_anova(df, ['space_score_mean'], Path('/tmp/test_out'))

print('Columns in anova_df:', anova_df.columns.tolist())
print('Columns in corr_df:', corr_df.columns.tolist())
print('Shape anova_df:', anova_df.shape)
print('Shape corr_df:', corr_df.shape)
print('\nFirst 3 rows of anova_df:')
print(anova_df.head(3).to_string())
print('\nFirst 3 rows of corr_df:')
print(corr_df.head(3).to_string())
