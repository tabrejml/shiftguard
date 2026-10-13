from __future__ import annotations
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, average_precision_score,
                            log_loss, f1_score)
from sklearn.model_selection import train_test_split, GroupShuffleSplit

UNSW_NUM = 'dur spkts dpkts sbytes dbytes rate sttl dttl sload dload sloss dloss sinpkt dinpkt sjit djit swin stcpb dtcpb dwin tcprtt synack ackdat smean dmean trans_depth response_body_len ct_srv_src ct_state_ttl ct_dst_ltm ct_src_dport_ltm ct_dst_sport_ltm ct_dst_src_ltm is_ftp_login ct_ftp_cmd ct_flw_http_mthd ct_src_ltm ct_srv_dst is_sm_ips_ports'.split()
UNSW_CAT = ['proto', 'service', 'state']
CANONICAL = 'duration_s src_packets dst_packets src_ip_bytes dst_ip_bytes mean_ip_len packets_per_s bytes_per_s'.split()
CIC_NUM = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length', 'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min']
SCHEMAS = {'unsw-nb15-v1': (UNSW_NUM, UNSW_CAT), 'canonical-pcap-v1': (CANONICAL, []), 'cicids2017-v1': (CIC_NUM, [])}
BENIGN = 'Benign'

def load_csv(path):
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    if df.columns.duplicated().any():
        raise ValueError('Duplicate column names after trimming.')
    return df

def features(df, schema):
    nums, cats = SCHEMAS[schema]
    missing = set(nums + cats) - set(df.columns)
    if missing:
        raise ValueError('Schema mismatch. Missing: ' + ', '.join(sorted(missing)))
    x = df[nums + cats].copy()
    for col in nums:
        raw = pd.to_numeric(x[col], errors='coerce')
        if ((raw.isna()) & (x[col].notna())).any():
            raise ValueError(f'{col}: nonnumeric values found.')
        if np.isinf(raw).any():
            raise ValueError(f'{col}: infinite values found; correct the extractor/data.')
        if (raw.dropna() < 0).any():
            raise ValueError(f'{col}: negative traffic measurements found.')
        x[col] = raw.astype(float)
    for col in cats:
        x[col] = x[col].fillna('missing').astype(str).str.strip().str.lower()
    if len(x) == 0:
        raise ValueError('No rows.')
    return x

def labels(df, target):
    if target == 'binary':
        if 'label' not in df:
            raise ValueError('Binary training/evaluation requires label (0 benign, 1 attack).')
        y = pd.to_numeric(df.label, errors='raise')
        if y.isna().any() or not set(y.unique()).issubset({0, 1}):
            raise ValueError('label must contain only 0 or 1.')
        return y.map({0: BENIGN, 1: 'Attack'}).to_numpy()
    if 'attack_cat' not in df or df.attack_cat.isna().any():
        raise ValueError('Category training requires nonmissing attack_cat.')
    y = df.attack_cat.astype(str).str.strip().replace({'Backdoor': 'Backdoors'})
    y = y.where(~y.str.lower().isin(['normal', 'benign']), BENIGN)
    if (y == '').any():
        raise ValueError('Empty category labels.')
    if 'label' in df and not np.array_equal(labels(df, 'binary') == BENIGN, y == BENIGN):
        raise ValueError('label and attack_cat disagree on benign/attack status.')
    return y.to_numpy()

def fingerprint(x):
    return pd.util.hash_pandas_object(x, index=False).to_numpy()

def audit_partitions(frames, schema, target, group=None):
    seen, groups = set(), set()
    report = {}
    for name, df in frames.items():
        x = features(df, schema)
        y = labels(df, target)
        hashes = set(fingerprint(x).tolist())
        overlap = seen & hashes
        if overlap:
            raise ValueError(f'{name}: {len(overlap)} duplicated feature signatures across partitions. Regroup/deduplicate before evaluation.')
        seen |= hashes
        if group:
            if group not in df or df[group].isna().any():
                raise ValueError(f'{name}: missing group metadata.')
            current = set(df[group].astype(str))
            if groups & current:
                raise ValueError(f'{name}: overlapping groups; split by group again.')
            groups |= current
        report[name] = {'rows': len(df), 'classes': pd.Series(y).value_counts().to_dict(),
                        'within_partition_duplicate_signatures': len(x) - len(hashes)}
    return report

def split_development(df, schema, target, mode='stratified', group=None, timestamp=None, seed=42):
    """Split by indivisible feature signatures, explicit groups, or chronological windows."""
    x = features(df, schema)
    y = labels(df, target)
    hashes = fingerprint(x)
    # Identical observations stay together; contradictory labels are rejected.
    sig = pd.DataFrame({'hash': hashes, 'label': y})
    if (sig.groupby('hash').label.nunique() > 1).any():
        raise ValueError('Identical feature rows have contradictory labels.')
    if mode == 'time':
        if not timestamp or timestamp not in df:
            raise ValueError('Time mode requires a timestamp column; row ID is not time.')
        t = pd.to_datetime(df[timestamp], utc=True, errors='raise')
        if t.isna().any():
            raise ValueError('Missing timestamps.')
        order = np.argsort(t.to_numpy(), kind='stable')
        n = len(df)
        # Boundary timestamps and repeated signatures/groups are purged from later windows.
        ranges = [order[:int(.6*n)], order[int(.6*n):int(.8*n)], order[int(.8*n):]]
        parts, seen, used_groups, last_t = {}, set(), set(), None
        for name, idx in zip(['train', 'calibration', 'validation'], ranges):
            keep = ~np.isin(hashes[idx], list(seen))
            if last_t is not None:
                keep &= (t.iloc[idx] > last_t).to_numpy()
            if group:
                if group not in df or df[group].isna().any():
                    raise ValueError('Missing group column.')
                keep &= ~df.iloc[idx][group].astype(str).isin(used_groups).to_numpy()
            idx = idx[keep]
            if len(idx) == 0:
                raise ValueError('Empty time partition after purging. Choose longer capture windows.')
            parts[name] = df.iloc[idx].copy()
            seen |= set(hashes[idx].tolist())
            last_t = t.iloc[idx].max()
            if group:
                used_groups |= set(df.iloc[idx][group].astype(str))
        return parts
    if group:
        if group not in df or df[group].isna().any():
            raise ValueError('Missing group metadata.')
        # Reject signatures spanning groups, otherwise explicit groups alone would leak.
        if pd.DataFrame({'hash': hashes, 'group': df[group].astype(str).to_numpy()}).groupby('hash')['group'].nunique().max() > 1:
            raise ValueError('Feature duplicates span groups. Deduplicate or use merged groups.')
        gids = df[group].astype(str).to_numpy()
    else:
        gids = hashes
    if mode == 'stratified' and not group:
        unique = sig.drop_duplicates('hash')
        a, b = train_test_split(unique['hash'], test_size=.4, stratify=unique.label, random_state=seed)
        rest = unique[unique['hash'].isin(b)]
        b, c = train_test_split(rest['hash'], test_size=.5, stratify=rest.label, random_state=seed)
        return {name: df.iloc[np.where(np.isin(hashes, ids))[0]].copy()
                for name, ids in zip(['train', 'calibration', 'validation'], [a, b, c])}
    a, rest = next(GroupShuffleSplit(n_splits=1, test_size=.4, random_state=seed).split(x, y, gids))
    b, c = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed+1).split(x.iloc[rest], y[rest], gids[rest]))
    return {name: df.iloc[idx].copy() for name, idx in zip(['train', 'calibration', 'validation'], [a, rest[b], rest[c]])}

def pipeline(schema, kind, seed=42):
    nums, cats = SCHEMAS[schema]
    transforms = [('numeric', Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                                      ('scale', StandardScaler())]), nums)]
    if cats:
        transforms.append(('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats))
    if kind == 'logistic':
        model = LogisticRegression(max_iter=2500, class_weight='balanced', random_state=seed)
    elif kind == 'forest':
        model = RandomForestClassifier(n_estimators=160, min_samples_leaf=2, class_weight='balanced', n_jobs=2, random_state=seed)
    else:
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=180, max_depth=5, learning_rate=.08, subsample=.85,
                              colsample_bytree=.85, tree_method='hist', n_jobs=2, random_state=seed)
    return Pipeline([('preprocess', ColumnTransformer(transforms)), ('model', model)])

def metrics(y, p, classes):
    y = np.asarray(y)
    classes = np.asarray(classes)
    pred = classes[p.argmax(axis=1)]
    attack = y != BENIGN
    ap = p[:, classes != BENIGN].sum(axis=1)
    benign_count = int((~attack).sum())
    per_class_ap = {}
    for j, c in enumerate(classes):
        z = y == c
        per_class_ap[str(c)] = float(average_precision_score(z, p[:, j])) if z.any() and (~z).any() else None
    known = np.isin(y, classes)
    confidence = p.max(axis=1)
    correct = pred == y
    ece = 0.
    for lo in np.linspace(0, .9, 10):
        m = (confidence >= lo) & (confidence < lo+.1 if lo < .9 else confidence <= 1)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - confidence[m].mean())
    labels_all = np.unique(np.r_[classes, y])
    return {'macro_f1': float(f1_score(y, pred, average='macro', zero_division=0)),
            'per_class': classification_report(y, pred, labels=labels_all, output_dict=True, zero_division=0),
            'attack_pr_auc_average_precision': float(average_precision_score(attack, ap)) if attack.any() and (~attack).any() else None,
            'per_class_average_precision': per_class_ap,
            'false_positive_rate': float(((pred != BENIGN) & ~attack).sum()/benign_count) if benign_count else None,
            'missed_attack_rate': float(((pred == BENIGN) & attack).sum()/attack.sum()) if attack.any() else None,
            'ece_10_bins': float(ece), 'attack_brier': float(np.mean((ap-attack.astype(float))**2)),
            'log_loss_known_classes': float(log_loss(y[known], p[known][:, np.argsort(classes)], labels=np.sort(classes))) if known.any() else None,
            'unseen_label_rows': int((~known).sum()),
            'confusion_labels': labels_all.tolist(),
            'confusion_matrix': confusion_matrix(y, pred, labels=labels_all).tolist()}

def review_curve(y, p, classes, thresholds=None):
    thresholds = np.linspace(0, 1, 101) if thresholds is None else thresholds
    classes = np.asarray(classes)
    pred = classes[p.argmax(axis=1)]
    attack = np.asarray(y) != BENIGN
    rows = []
    for threshold in thresholds:
        review = p.max(axis=1) < threshold
        accepted = ~review
        missed = attack & (pred == BENIGN) & accepted
        rows.append({'threshold': float(threshold), 'review_fraction': float(review.mean()),
                     'review_count': int(review.sum()), 'auto_missed_attacks': int(missed.sum()),
                     'auto_missed_attack_rate': float(missed.sum()/attack.sum()) if attack.any() else None,
                     'coverage': float(accepted.mean()),
                     'accepted_error': float((pred[accepted] != np.asarray(y)[accepted]).mean()) if accepted.any() else None})
    return pd.DataFrame(rows)

def profile(x):
    result = {}
    for c in x:
        if pd.api.types.is_numeric_dtype(x[c]):
            values = x[c].dropna().to_numpy()
            edges = np.unique(np.quantile(values, np.linspace(0, 1, 11))) if len(values) else np.array([0., 1.])
            edges = np.r_[-np.inf, edges[1:-1], np.inf]
            if len(edges) < 3:
                center = values[0] if len(values) else 0.
                edges = np.array([-np.inf, center-.5, center+.5, np.inf])
            counts = np.histogram(values, edges)[0]
            probs = (counts+.5)/(counts.sum()+.5*len(counts))
            result[c] = {'type': 'numeric', 'edges': edges.tolist(), 'probs': probs.tolist(),
                         'median': float(np.nanmedian(values)) if len(values) else 0., 'missing': float(x[c].isna().mean())}
        else:
            counts = x[c].value_counts()
            result[c] = {'type': 'category', 'probs': (counts/len(x)).to_dict(),
                         'mode': str(counts.index[0]), 'missing': 0.}
    return result

def drift(x, reference):
    rows = []
    for c, r in reference.items():
        if r['type'] == 'numeric':
            counts = np.histogram(x[c].dropna(), r['edges'])[0]
            q = (counts+.5)/(counts.sum()+.5*len(counts))
            p = np.array(r['probs'])
            score = float(np.sum((q-p)*np.log(q/p)))
            metric = 'PSI'
            unseen = None
        else:
            qd = x[c].value_counts(normalize=True).to_dict()
            keys = set(r['probs']) | set(qd)
            score = .5 * sum(abs(r['probs'].get(k, 0)-qd.get(k, 0)) for k in keys)
            metric = 'Total variation'
            unseen = float((~x[c].isin(r['probs'])).mean())
        rows.append({'feature': c, 'metric': metric, 'shift_score': score,
                     'missing_fraction': float(x[c].isna().mean()), 'unseen_category_fraction': unseen})
    return pd.DataFrame(rows).sort_values('shift_score', ascending=False)

def score(bundle, df, threshold=None):
    x = features(df, bundle['schema'])
    start = time.perf_counter()
    p = bundle['model'].predict_proba(x)
    elapsed = (time.perf_counter()-start)*1000
    classes = np.asarray(bundle['classes'])
    pred = classes[p.argmax(axis=1)]
    confidence = p.max(axis=1)
    threshold = bundle['threshold'] if threshold is None else threshold
    out = pd.DataFrame({'flow_id': df['flow_id'].astype(str).to_numpy() if 'flow_id' in df else [f'row-{i}' for i in range(len(df))],
                        'prediction': pred, 'confidence': confidence,
                        'attack_probability': p[:, classes != BENIGN].sum(axis=1),
                        'entropy': -(p*np.log(np.clip(p, 1e-12, 1))).sum(axis=1)/np.log(len(classes)),
                        'review_required': confidence < threshold,
                        'demo': bool(bundle.get('demo', False)), 'schema': bundle['schema'],
                        'model_kind': bundle['kind'], 'threshold_used': float(threshold),
                        'frozen_review_threshold': float(bundle['threshold'])})
    for c in ['timestamp', 'src_ip', 'dst_ip', 'device_id', 'attack_cat', 'label', 'capture_day', 'source_file', 'source_row']:
        if c in df:
            out[c] = df[c].to_numpy()
    return out, p, elapsed

def explain(bundle, row):
    """Raw-feature reference replacement; not SHAP, additive attribution, or causation."""
    x = features(row, bundle['schema'])
    p = bundle['model'].predict_proba(x)[0]
    j = int(p.argmax())
    variants = pd.concat([x]*len(x.columns), ignore_index=True)
    observations = []
    for i, c in enumerate(x.columns):
        r = bundle['reference'][c]
        ref = r['median'] if r['type'] == 'numeric' else r['mode']
        variants.loc[i, c] = ref
        observations.append({'feature': c, 'observed': str(x.iloc[0][c]), 'reference': str(ref)})
    replaced = bundle['model'].predict_proba(variants)[:, j]
    for o, value in zip(observations, replaced):
        o['probability_change'] = float(p[j]-value)
    return pd.DataFrame(observations).assign(magnitude=lambda d: d.probability_change.abs()).sort_values('magnitude', ascending=False)

def train(frames, output, schema='unsw-nb15-v1', target='binary', group=None, budget=.15, seed=42, kinds=None):
    import joblib
    import sklearn
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    audit = audit_partitions(frames, schema, target, group)
    xs = {k: features(v, schema) for k, v in frames.items()}
    ys = {k: labels(v, target) for k, v in frames.items()}
    classes = np.unique(ys['train'])
    if BENIGN not in classes or len(classes) < 2:
        raise ValueError('Training needs benign and at least one attack class.')
    for k in ['calibration', 'validation']:
        if set(ys[k]) != set(classes):
            raise ValueError(f'{k}: every training class must be represented; resplit with sufficient records.')
    encoded = {c: i for i, c in enumerate(classes)}
    summary, fitted = {}, {}
    for kind in kinds or ['logistic', 'forest', 'xgboost']:
        print(f'Training {kind}: fit={len(xs["train"]):,}, calibration={len(xs["calibration"]):,}, validation={len(xs["validation"]):,}, test={len(xs["test"]):,}', flush=True)
        base = pipeline(schema, kind, seed)
        t = time.perf_counter()
        # XGBoost uses integer labels; the wrapper consistently calibrates all models in that label space.
        base.fit(xs['train'], np.array([encoded[c] for c in ys['train']]))
        cal = CalibratedClassifierCV(FrozenEstimator(base), method='sigmoid')
        cal.fit(xs['calibration'], np.array([encoded[c] for c in ys['calibration']]))
        fit_seconds = time.perf_counter()-t
        val_p = cal.predict_proba(xs['validation'])
        val_metrics = metrics(ys['validation'], val_p, classes)
        curve = review_curve(ys['validation'], val_p, classes)
        feasible = curve[curve.review_fraction <= budget]
        chosen = feasible.sort_values(['auto_missed_attacks', 'accepted_error', 'threshold'], ascending=[True, True, True], na_position='last').iloc[0]
        start = time.perf_counter()
        test_p = cal.predict_proba(xs['test'])
        latency = (time.perf_counter()-start)*1000
        raw_test_p = base.predict_proba(xs['test'])
        summary[kind] = {'validation': val_metrics, 'test': metrics(ys['test'], test_p, classes),
                         'uncalibrated_test_ablation': metrics(ys['test'], raw_test_p, classes),
                         'selected_threshold': float(chosen.threshold), 'validation_review_fraction': float(chosen.review_fraction),
                         'validation_budget': budget, 'test_batch_latency_ms': latency,
                         'test_amortized_ms_per_flow': latency/len(xs['test']), 'fit_and_calibration_seconds': fit_seconds}
        curve.to_csv(output/f'{kind}_validation_review_curve.csv', index=False)
        review_curve(ys['test'], test_p, classes).to_csv(output/f'{kind}_test_review_curve.csv', index=False)
        bundle = {'model': cal, 'schema': schema, 'target': target, 'classes': classes.tolist(),
                  'threshold': float(chosen.threshold), 'reference': profile(xs['train']), 'kind': kind,
                  'seed': seed, 'validation_budget': budget, 'metrics': summary[kind]}
        joblib.dump(bundle, output/f'{kind}.joblib')
        fitted[kind] = bundle
    winner = max(summary, key=lambda k: summary[k]['validation']['macro_f1'])
    joblib.dump(fitted[winner], output/'selected.joblib')
    manifest = {'selected_on': 'validation macro-F1; ties follow model order', 'selected_model': winner,
                'schema': schema, 'target': target, 'seed': seed, 'partition_audit': audit,
                'python': platform.python_version(), 'sklearn': sklearn.__version__, 'models': summary,
                'input_sha256': {k: hashlib.sha256(v.to_csv(index=False).encode()).hexdigest() for k, v in frames.items()}}
    (output/'evaluation.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return manifest
