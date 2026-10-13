import argparse
import json
from pathlib import Path
import joblib
from .core import SCHEMAS, load_csv, split_development, train, score, labels, metrics, features, drift, review_curve, explain

def main():
    p = argparse.ArgumentParser(description='ShiftGuard reproducible IDS workflow')
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('split')
    s.add_argument('--source', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--mode', choices=['stratified', 'group', 'time'], default='stratified')
    s.add_argument('--group')
    s.add_argument('--timestamp')
    s.add_argument('--schema', choices=SCHEMAS, default='unsw-nb15-v1')
    s.add_argument('--target', choices=['binary', 'category'], default='binary')
    s.add_argument('--seed', type=int, default=42)
    t = sub.add_parser('train')
    for name in ['train', 'calibration', 'validation', 'test']:
        t.add_argument('--'+name, required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--schema', choices=SCHEMAS, default='unsw-nb15-v1')
    t.add_argument('--target', choices=['binary', 'category'], default='binary')
    t.add_argument('--group')
    t.add_argument('--budget', type=float, default=.15)
    t.add_argument('--seed', type=int, default=42)
    q = sub.add_parser('score')
    q.add_argument('--model', required=True)
    q.add_argument('--input', required=True)
    q.add_argument('--out', required=True)
    q.add_argument('--explain-limit', type=int, default=0, help='Include reference-replacement explanations for first N rows')
    e = sub.add_parser('evaluate')
    e.add_argument('--model', required=True)
    e.add_argument('--input', required=True)
    e.add_argument('--out', required=True)
    args = p.parse_args()
    if args.command == 'split':
        if args.mode == 'group' and not args.group:
            p.error('group mode requires --group')
        parts = split_development(load_csv(args.source), args.schema, args.target, args.mode, args.group, args.timestamp, args.seed)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        for name, frame in parts.items():
            frame.to_csv(Path(args.out)/f'{name}.csv', index=False)
        print(json.dumps({k: len(v) for k, v in parts.items()}))
    elif args.command == 'train':
        if not 0 <= args.budget <= 1:
            p.error('budget must be between 0 and 1')
        frames = {k: load_csv(getattr(args, k)) for k in ['train', 'calibration', 'validation', 'test']}
        result = train(frames, args.out, args.schema, args.target, args.group, args.budget, args.seed)
        print('Selected on validation:', result['selected_model'])
    else:
        # Only load model artifacts you trained/trust; joblib is executable serialization.
        bundle = joblib.load(args.model)
        frame = load_csv(args.input)
        if bundle['schema'] == 'cicids2017-v1':
            from .datasets import normalize_cic
            frame, audit = normalize_cic(frame, Path(args.input).name)
            print(f'CIC adapter: {audit["raw_rows"]:,} input rows; {audit["kept_rows"]:,} retained.')
        scored, probs, latency = score(bundle, frame)
        if args.command == 'score':
            if args.explain_limit > 0:
                scored['explanation_json'] = ''
                for i in range(min(len(frame), args.explain_limit)):
                    scored.loc[i, 'explanation_json'] = explain(bundle, frame.iloc[[i]]).drop(columns='magnitude').to_json(orient='records')
            scored.to_csv(args.out, index=False)
            print(f'{len(scored)} flows; batch latency {latency:.2f} ms')
        else:
            y = labels(frame, bundle['target'])
            report = metrics(y, probs, bundle['classes'])
            report['review_curve'] = json.loads(review_curve(y, probs, bundle['classes']).to_json(orient='records'))
            report['drift'] = json.loads(drift(features(frame, bundle['schema']), bundle['reference']).to_json(orient='records'))
            report['batch_latency_ms'] = latency
            Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False))

if __name__ == '__main__':
    main()
