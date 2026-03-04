#!/usr/bin/env python
import pandas as pd
from feature_pipeline.utils.data_loader import load_feature_dataset
import pingouin as pg

X, y, team, df = load_feature_dataset()

anova_table = pg.anova(
    data=df,
    dv='space_score_mean',
    between=['team_lost_possession', 'label'],
    detailed=True,
)

with open('/tmp/anova_output.txt', 'w') as f:
    f.write("ANOVA output columns:\n")
    f.write(str(anova_table.columns.tolist()) + "\n\n")
    f.write("ANOVA table:\n")
    f.write(anova_table.to_string())
