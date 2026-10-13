import sqlite3
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

def connect(path='runtime/reviews.sqlite'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY, dataset TEXT, model TEXT, row_key TEXT, flow_id TEXT, decision TEXT, analyst TEXT, notes TEXT, created TEXT)')
    return db

def save_review(dataset, model, row_key, flow_id, decision, analyst, notes, path='runtime/reviews.sqlite'):
    if decision not in ['Benign', 'Attack', 'Unresolved']:
        raise ValueError('Invalid review decision')
    if not analyst.strip():
        raise ValueError('Analyst name/ID required.')
    with connect(path) as db:
        db.execute('INSERT INTO reviews (dataset,model,row_key,flow_id,decision,analyst,notes,created) VALUES (?,?,?,?,?,?,?,?)',
                   (dataset, model, str(row_key), str(flow_id), decision, analyst.strip(), notes,
                    datetime.now(timezone.utc).isoformat()))

def read_reviews(dataset, model, path='runtime/reviews.sqlite'):
    with connect(path) as db:
        return pd.read_sql_query('SELECT * FROM reviews WHERE dataset=? AND model=? ORDER BY id', db, params=(dataset, model))
