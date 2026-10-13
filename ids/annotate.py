"""Join independently verified labels to extracted flows by exact flow_id."""
import argparse
from .core import load_csv, labels

def join(flows, annotations):
    if 'flow_id' not in flows or 'flow_id' not in annotations:
        raise ValueError('Both inputs need flow_id.')
    for df in [flows, annotations]:
        if df.flow_id.isna().any() or df.flow_id.duplicated().any():
            raise ValueError('flow_id must be unique and nonmissing.')
    if 'label' not in annotations:
        raise ValueError('Annotations need label: 0 benign, 1 attack.')
    if 'label' in flows or 'attack_cat' in flows:
        raise ValueError('Flows already contain labels; do not overwrite ground truth silently.')
    cols = ['flow_id', 'label'] + (['attack_cat'] if 'attack_cat' in annotations else [])
    merged = flows.merge(annotations[cols], on='flow_id', how='left', validate='one_to_one')
    if merged.label.isna().any():
        raise ValueError('Some extracted flows lack verified labels. Do not assume unlabelled means benign.')
    if not set(annotations.flow_id).issubset(set(flows.flow_id)):
        raise ValueError('Annotations contain unknown flow IDs.')
    labels(merged, 'category' if 'attack_cat' in merged else 'binary')
    return merged

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--flows', required=True)
    p.add_argument('--annotations', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    result = join(load_csv(a.flows), load_csv(a.annotations))
    result.to_csv(a.out, index=False)
    print(f'{len(result)} independently labelled flows written.')

if __name__ == '__main__':
    main()
