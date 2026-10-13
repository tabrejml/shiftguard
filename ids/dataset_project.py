"""Reproduce the measured two-dataset project from the supplied CSV archives."""
import argparse
import hashlib
import json
import gc
import shutil
import zipfile
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from .core import (load_csv, features, fingerprint, split_development, train, score, labels,
                   metrics, review_curve, drift, explain, CIC_NUM)
from .datasets import normalize_cic

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as fp:
        for chunk in iter(lambda:fp.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def write_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))

def extract_archive(path,destination):
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        files=[item for item in archive.infolist() if not item.is_dir()]
        if sum(item.file_size for item in files)>2*1024**3:
            raise ValueError('Archive exceeds the 2 GiB uncompressed cap.')
        names=[Path(item.filename).name for item in files]
        if len(names)!=len(set(names)):
            raise ValueError('Archive has colliding flattened filenames.')
        for item,name in zip(files,names):
            if not name.lower().endswith('.csv'):
                raise ValueError('Only dataset CSV members are supported.')
            target=destination/name
            if target.exists():
                raise ValueError(f'{target} already exists. Keep the existing data or extract into a fresh directory.')
            with archive.open(item) as source,open(target,'wb') as output:
                shutil.copyfileobj(source,output)

def ambiguous(hashes, y):
    df=pd.DataFrame({'hash':hashes,'label':y})
    counts=df.groupby('hash').label.nunique()
    return set(counts[counts>1].index)

def prepare_unsw(root):
    raw=root/'raw';prepared=root/'prepared';prepared.mkdir(exist_ok=True)
    inventory=[]
    for path in sorted(raw.glob('UNSW-NB15_[1-4].csv')):
        count=0;classes=Counter();start=None;end=None
        for df in pd.read_csv(path,header=None,usecols=[28,29,47,48],encoding='utf-8-sig',chunksize=100000,low_memory=False):
            count+=len(df);cat=df[47].fillna('Normal').astype(str).str.strip().replace({'Backdoor':'Backdoors','Backdoors':'Backdoors'})
            classes.update(cat.value_counts().to_dict())
            lo=pd.to_numeric(df[28],errors='raise').min();hi=pd.to_numeric(df[29],errors='raise').max()
            start=float(lo) if start is None else min(start,float(lo));end=float(hi) if end is None else max(end,float(hi))
        inventory.append({'file':path.name,'rows':count,'attack_categories':dict(classes),'first_epoch':start,'last_epoch':end,'sha256':digest(path),'role':'raw archive inventory only; not pooled with overlapping curated subsets'})
        print('Audited',path.name,count,flush=True)
    source=load_csv(raw/'UNSW_NB15_training-set.csv');test=load_csv(raw/'UNSW_NB15_testing-set.csv')
    for name,df in [('UNSW_NB15_training-set.csv',source),('UNSW_NB15_testing-set.csv',test)]:
        df['flow_id']=[f'UNSW:{name}:{i}' for i in df.id]
        df['source_file']=name;df['source_row']=np.arange(len(df))+2
        inventory.append({'file':name,'rows':len(df),'attack_categories':df.attack_cat.value_counts().to_dict(),'sha256':digest(raw/name),'role':'development source' if df is source else 'held-out test'})
    hs=fingerprint(features(source,'unsw-nb15-v1'));ht=fingerprint(features(test,'unsw-nb15-v1'))
    conflict=ambiguous(hs,source.label)
    conflicting_test=ambiguous(ht,test.label)
    test_drop=np.isin(ht,list(conflicting_test));test=test.loc[~test_drop].copy();ht=ht[~test_drop]
    cross=np.isin(hs,ht);bad=np.isin(hs,list(conflict))
    kept=source.loc[~(cross|bad)].copy()
    parts=split_development(kept,'unsw-nb15-v1','binary',seed=42);parts['test']=test
    for name,df in parts.items():df.to_pickle(prepared/f'{name}.pkl')
    audit={'dataset':'UNSW-NB15','archive':'archive (4).zip','schema':'unsw-nb15-v1','predictors':42,'files':inventory,
           'split':'supplied filename training source split by feature signatures 60/20/20; supplied filename testing held out',
           'filename_caution':'Supplied training has 82,332 rows and testing has 175,341; opposite to the conventional published sizes. Filenames are preserved, not silently swapped.',
           'development_raw_rows':len(source),'test_raw_rows':int(len(test)+test_drop.sum()),
           'development_rows_matching_test_removed':int(cross.sum()),'development_binary_conflict_rows_removed':int(bad.sum()),
           'test_binary_conflict_rows_removed':int(test_drop.sum()),'development_retained':len(kept),
           'limitations':['This is the supplied filename split, not the conventional published direction.','Curated subsets have no timestamps or endpoints; raw archive metadata was inventoried only.','No cross-dataset transfer was run.']}
    write_json(root/'audit.json',audit);return audit

def prepare_cic(root, cap=200000):
    raw=root/'raw';prepared=root/'prepared';prepared.mkdir(exist_ok=True)
    inventory=[];test_parts=[];reservoir=None;rng=np.random.default_rng(42)
    source_hashes=[];source_labels=[]
    for path in sorted(raw.glob('*.csv')):
        stats=Counter();classes=Counter();clean_classes=Counter();is_test=path.name.startswith('Friday')
        for frame in pd.read_csv(path,encoding='cp1252',chunksize=50000,low_memory=False):
            frame.columns=frame.columns.str.strip()
            classes.update(frame['Label'].astype(str).str.strip().value_counts().to_dict())
            clean,chunk_stats=normalize_cic(frame,path.name);stats.update(chunk_stats)
            clean_classes.update(clean.attack_cat.value_counts().to_dict())
            if is_test:
                test_parts.append(clean)
            else:
                source_hashes.append(fingerprint(features(clean,'cicids2017-v1')));source_labels.append(clean.label.to_numpy())
                clean['_sample_priority']=rng.random(len(clean))
                reservoir=pd.concat([reservoir,clean],ignore_index=True) if reservoir is not None else clean
                if len(reservoir)>cap:reservoir=reservoir.nsmallest(cap,'_sample_priority').copy()
        inventory.append({'file':path.name,**dict(stats),'raw_attack_categories':dict(classes),'clean_attack_categories':dict(clean_classes),
                          'role':'held-out Friday test' if is_test else 'Monday-Thursday development pool','sha256':digest(path)})
        print('Audited',path.name,stats['raw_rows'],'kept',stats['kept_rows'],flush=True)
    test=pd.concat(test_parts,ignore_index=True);del test_parts
    ht=fingerprint(features(test,'cicids2017-v1'))
    test_conflicts=ambiguous(ht,test.label)
    td=np.isin(ht,list(test_conflicts));test=test.loc[~td].copy();ht=ht[~td]
    all_h=np.concatenate(source_hashes);all_y=np.concatenate(source_labels);del source_hashes,source_labels
    conflicts=ambiguous(all_h,all_y)
    source_cross=int(np.isin(all_h,ht).sum());source_bad=int(np.isin(all_h,list(conflicts)).sum())
    del all_h,all_y;gc.collect()
    reservoir=reservoir.drop(columns='_sample_priority')
    hs=fingerprint(features(reservoir,'cicids2017-v1'));cross=np.isin(hs,ht);bad=np.isin(hs,list(conflicts))
    kept=reservoir.loc[~(cross|bad)].copy()
    parts=split_development(kept,'cicids2017-v1','binary',seed=42);parts['test']=test
    for name,df in parts.items():df.to_pickle(prepared/f'{name}.pkl')
    audit={'dataset':'CICIDS2017','archive':'archive (3).zip','schema':'cicids2017-v1','predictors':len(CIC_NUM),'files':inventory,
           'split':'uniform seeded reservoir of up to 200,000 valid Monday-Thursday rows split by feature signatures 60/20/20; all valid Friday flows held out',
           'raw_rows':sum(f['raw_rows'] for f in inventory),'clean_rows':sum(f['kept_rows'] for f in inventory),
           'development_pool_clean_rows':sum(f['kept_rows'] for f in inventory if not f['file'].startswith('Friday')),
           'development_sample_before_audit':len(reservoir),'development_retained':len(kept),
           'development_pool_rows_matching_test':source_cross,'development_pool_binary_conflict_rows':source_bad,
           'sample_rows_matching_test_removed':int(cross.sum()),'sample_binary_conflict_rows_removed':int(bad.sum()),
           'test_raw_clean_rows':int(len(test)+td.sum()),'test_binary_conflict_rows_removed':int(td.sum()),
           'cleaning':['Trim column names; verify duplicate Fwd Header Length and use one copy.','Exclude Destination Port from predictors.',
                       'Convert infinite numeric measurements and -1 initial-window sentinels to missing; impute using fit-partition medians.',
                       'Drop other negative measurement rows and missing labels.','Remove binary-label-conflicting signatures and development signatures shared with the Friday test.'],
           'limitations':['Development is a 200,000-row maximum uniform sample, not full-data training.','Friday is day-held-out; CSVs lack timestamps, endpoints and session IDs.',
                          'Friday categories include DDoS, PortScan and Bot, absent from the Monday-Thursday category pool. Binary detection is tested, not category prediction or real zero-day detection.',
                          'No UNSW-to-CIC transfer is claimed; feature schemas differ.']}
    write_json(root/'audit.json',audit);return audit

def fit_dataset(root,key):
    frames={k:pd.read_pickle(root/'prepared'/f'{k}.pkl') for k in ['train','calibration','validation','test']}
    audit=json.loads((root/'audit.json').read_text());out=root/'results'
    result=train(frames,out,schema=audit['schema'],target='binary',budget=.15,seed=42)
    reference=joblib.load(out/'selected.joblib')['reference']
    result['audit']=audit;result['demo']=False;result['dataset_key']=key
    result['drift']=json.loads(drift(features(frames['test'],audit['schema']),reference).to_json(orient='records'))
    result['test_categories']=frames['test'].attack_cat.value_counts().to_dict()
    result['development_categories']=pd.concat([frames[k].attack_cat for k in ['train','calibration','validation']]).value_counts().to_dict()
    result['unseen_test_categories']=sorted(set(result['test_categories'])-set(result['development_categories']))
    comparison=[]
    for kind,record in result['models'].items():
        bundle=joblib.load(out/f'{kind}.joblib')
        bundle.update({'demo':False,'dataset_key':key,'dataset_name':audit['dataset'],'data_origin':'uploaded public dataset CSVs',
                       'default_flows':f'datasets/{key}/dashboard_flows.csv'})
        scored,p,elapsed=score(bundle,frames['test'])
        policy=json.loads(review_curve(labels(frames['test'],'binary'),p,bundle['classes'],[bundle['threshold']]).to_json(orient='records'))[0]
        policy['attention_count']=int(((scored.prediction!='Benign')|scored.review_required).sum())
        policy['attention_fraction']=policy['attention_count']/len(scored)
        record['frozen_test_policy']=policy
        pred=scored.prediction.to_numpy();review=scored.review_required.to_numpy()
        record['attack_category_binary_detection']=[]
        for category,idx in frames['test'].groupby('attack_cat').indices.items():
            benign=str(category).lower() in ['normal','benign']
            record['attack_category_binary_detection'].append({'category':str(category),'rows':len(idx),
                'correct_binary_fraction':float(((pred[idx]=='Benign') if benign else (pred[idx]!='Benign')).mean()),
                'review_fraction':float(review[idx].mean()),'automatic_missed_attacks':int(((pred[idx]=='Benign')&~review[idx]).sum()) if not benign else 0})
        bundle['metrics']=record;joblib.dump(bundle,out/f'{kind}.joblib')
        model_dir=root.parent.parent/'models';model_dir.mkdir(exist_ok=True)
        joblib.dump(bundle,model_dir/f'{key}_{kind}.joblib')
        if kind==result['selected_model']:
            joblib.dump(bundle,out/'selected.joblib')
            sample=frames['test'].sample(n=min(5000,len(frames['test'])),random_state=42).copy().reset_index(drop=True)
            sample.to_csv(root/'dashboard_flows.csv',index=False)
            display,_,_=score(bundle,sample)
            display['explanation_json']=''
            for i in range(min(25,len(sample))):
                display.loc[i,'explanation_json']=explain(bundle,sample.iloc[[i]]).drop(columns='magnitude').to_json(orient='records')
            display.to_csv(out/'dashboard_predictions.csv',index=False)
            result['dashboard_sample']={'rows':len(sample),'selection':'uniform test sample, seed 42; dashboard figures apply to this sample only',
                                        'explained_rows':min(25,len(sample))}
            # Bin-level reliability is preserved for a publication plot.
            c=p.max(axis=1);correct=pred==labels(frames['test'],'binary');reliability=[]
            for lo in np.arange(0,1,.1):
                mask=(c>=lo)&((c<lo+.1) if lo<.9 else (c<=1))
                if mask.any():reliability.append({'bin_lower':float(lo),'rows':int(mask.sum()),'mean_confidence':float(c[mask].mean()),'accuracy':float(correct[mask].mean())})
            result['reliability']=reliability
        comparison.append({'model':kind,'selected':kind==result['selected_model'],'validation_macro_f1':record['validation']['macro_f1'],
                           'test_macro_f1':record['test']['macro_f1'],'test_attack_average_precision':record['test']['attack_pr_auc_average_precision'],
                           'test_fpr':record['test']['false_positive_rate'],'test_attack_recall':record['test']['per_class']['Attack']['recall'],
                           'test_ece':record['test']['ece_10_bins'],'review_threshold':record['selected_threshold'],
                           'test_review_fraction':policy['review_fraction'],'test_auto_missed_attack_rate':policy['auto_missed_attack_rate']})
        del scored,p;gc.collect()
    pd.DataFrame(comparison).to_csv(out/'baseline_comparison.csv',index=False)
    write_json(out/'dataset_report.json',result)
    print(key,'COMPLETE selected=',result['selected_model'],flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='datasets')
    p.add_argument('--dataset',choices=['unsw_nb15','cicids2017','both'],default='both')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--fit-only',action='store_true')
    p.add_argument('--unsw-zip');p.add_argument('--cic-zip')
    a=p.parse_args();root=Path(a.root)
    if a.unsw_zip:extract_archive(a.unsw_zip,root/'unsw_nb15/raw')
    if a.cic_zip:extract_archive(a.cic_zip,root/'cicids2017/raw')
    for key in (['unsw_nb15','cicids2017'] if a.dataset=='both' else [a.dataset]):
        path=root/key
        if not a.fit_only:
            prepare_unsw(path) if key=='unsw_nb15' else prepare_cic(path)
        if not a.prepare_only:fit_dataset(path,key)

if __name__=='__main__':main()
