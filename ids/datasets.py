"""Dataset-specific measurement contracts, never a UNSW-to-CIC CSV adapter."""
import re
import numpy as np
import pandas as pd
from .core import CIC_NUM, features

def normalize_cic(df, source='uploaded.csv'):
    df = df.copy()
    df.columns = df.columns.str.strip()
    if df.columns.duplicated().any():
        raise ValueError('Duplicate columns after trimming.')
    missing = set(CIC_NUM)-set(df.columns)
    if missing:
        raise ValueError('CIC schema mismatch: '+', '.join(sorted(missing)))
    stats = {'raw_rows': len(df)}
    if 'Fwd Header Length.1' in df:
        left = pd.to_numeric(df['Fwd Header Length'], errors='raise')
        right = pd.to_numeric(df['Fwd Header Length.1'], errors='raise')
        stats['duplicate_header_column_disagreements'] = int((~(left.eq(right)|(left.isna()&right.isna()))).sum())
        if stats['duplicate_header_column_disagreements']:
            raise ValueError('Repeated Fwd Header Length columns disagree.')
    numeric = df[CIC_NUM].apply(pd.to_numeric, errors='raise').astype(float)
    stats['original_missing_cells'] = int(numeric.isna().sum().sum())
    inf = np.isinf(numeric.to_numpy())
    stats['infinite_cells_to_missing'] = int(inf.sum())
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    stats['minus_one_window_sentinels_to_missing'] = 0
    for c in ['Init_Win_bytes_forward', 'Init_Win_bytes_backward']:
        sentinel = numeric[c] == -1
        stats['minus_one_window_sentinels_to_missing'] += int(sentinel.sum())
        numeric.loc[sentinel, c] = np.nan
    negative = (numeric < 0).any(axis=1)
    stats['negative_measurement_rows_dropped'] = int(negative.sum())
    if 'Label' in df:
        category = df['Label'].astype('string').str.strip()
    elif 'attack_cat' in df:
        category = df['attack_cat'].astype('string').str.strip()
    else:
        category = None
    if category is not None:
        category = category.str.replace('ï¿½', ' - ', regex=False)
        category = category.str.replace(r'\s*[\x96\u2013\u2014\ufffd]\s*', ' - ', regex=True)
        category = category.str.replace(r'\s+', ' ', regex=True)
        category = category.where(~category.str.lower().isin(['benign', 'normal']), 'Benign')
        invalid_label = category.isna()|category.eq('')
    else:
        invalid_label = pd.Series(False, index=df.index)
    keep = ~(negative | invalid_label)
    stats['missing_label_rows_dropped'] = int(invalid_label.sum())
    stats['kept_rows'] = int(keep.sum())
    out = numeric.loc[keep].copy()
    if category is not None:
        out['attack_cat'] = category.loc[keep].astype(str)
        out['label'] = (out.attack_cat != 'Benign').astype(int)
    elif 'label' in df:
        out['label'] = df.loc[keep, 'label']
    out['source_file'] = df.loc[keep, 'source_file'] if 'source_file' in df else source
    out['source_row'] = df.loc[keep, 'source_row'] if 'source_row' in df else df.loc[keep].index+2
    match = re.search(r'(Monday|Tuesday|Wednesday|Thursday|Friday)', source, re.I)
    out['capture_day'] = df.loc[keep, 'capture_day'] if 'capture_day' in df else (match.group(1).capitalize() if match else 'Unknown')
    out['flow_id'] = df.loc[keep, 'flow_id'] if 'flow_id' in df else [f'CIC:{source}:{i}' for i in out.source_row]
    features(out, 'cicids2017-v1')
    return out.reset_index(drop=True), stats
