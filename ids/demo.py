import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from .core import pipeline, profile, CANONICAL

def synthetic(n=400, seed=42):
    r = np.random.default_rng(seed)
    latent = r.normal(0, 1, n)
    duration = np.exp(r.normal(.2, 1.3, n))
    src = np.maximum(1, (np.exp(3 + .7*latent + r.normal(0, .8, n))).astype(int))
    dst = np.maximum(0, (src*r.uniform(.05, 1.4, n)).astype(int))
    sb = src*r.uniform(60, 1200, n)
    db = dst*r.uniform(60, 1200, n)
    df = pd.DataFrame({'duration_s': duration, 'src_packets': src, 'dst_packets': dst,
                       'src_ip_bytes': sb, 'dst_ip_bytes': db,
                       'mean_ip_len': (sb+db)/(src+dst), 'packets_per_s': (src+dst)/duration,
                       'bytes_per_s': (sb+db)/duration})
    df['label'] = (latent+r.normal(0, .9, n) > .1).astype(int)
    df['flow_id'] = [f'synthetic-{seed}-{i:05}' for i in range(n)]
    df['timestamp'] = pd.date_range('2026-01-01T08:00:00Z', periods=n, freq='15s').astype(str)
    df['src_ip'] = [f'192.0.2.{i%20+1}' for i in range(n)]
    df['dst_ip'] = [f'198.51.100.{i%5+1}' for i in range(n)]
    return df

def demo_bundle():
    fit = synthetic(1600, 9)
    calibration = synthetic(500, 10)
    model = pipeline('canonical-pcap-v1', 'logistic')
    model.fit(fit[CANONICAL], fit.label)
    calibrated = CalibratedClassifierCV(FrozenEstimator(model), method='sigmoid')
    calibrated.fit(calibration[CANONICAL], calibration.label)
    return {'model': calibrated, 'schema': 'canonical-pcap-v1', 'target': 'binary',
            'classes': ['Benign', 'Attack'], 'threshold': .75, 'reference': profile(fit[CANONICAL]),
            'kind': 'synthetic-demo-logistic', 'metrics': {}, 'demo': True}
