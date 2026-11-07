"""Task 4: frozen HGB. Run from the project root (or pass --root). Never uploads anything."""
import argparse
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits

PARAMS = dict(max_leaf_nodes=63, min_samples_leaf=30, learning_rate=0.04, max_iter=400,
              l2_regularization=1, loss="absolute_error", categorical_features=["suburb", "property_type"],
              early_stopping=False, random_state=42)
FEATURES = ["prior_flat_rent", "days_since_ad", "building_rent_365d", "building_n_365d"]
BUILDING = ["suburb", "building_address", "bedrooms"]
RAW_NUM = ["bedrooms", "bathrooms", "car_space", "latitude", "longitude"]
CATS = ["suburb","property_type"]
BASE = ["gnaf_id","building_address",*CATS,*RAW_NUM]
KEY = ["gnaf_id","listing_date"]
SUB = ["gnaf_id","rent_with_car_space","rent_without_car_space"]
CUTOFF = pd.Timestamp("2026-07-31")
HASH_MAR = "0beb0a15e47e1aded0f15d1ad525fdfa37f3ad347cf49f0a3085ff05b78fd178"

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load(root):
    data = root/"data"
    raw_flats = data/"raw"/"rents-input-flats-city-of-sydney.csv"
    raw_ads = data/"raw"/"rents-past-leases-city-of-sydney.csv"
    processed = data/"processed"
    fv, av = processed/"flats_clean_transit_v2.csv", processed/"leases_clean_transit_v2.csv"
    fc, ac = processed/"flats_clean.csv", processed/"leases_clean.csv"
    paths = [(fv, av, "v2 clean tables, distances ignored"),
             (fc, ac, "base clean tables"), (raw_flats, raw_ads, "raw with model-column cleaning")]
    a,b,origin = next((a,b,tag) for a,b,tag in paths if a.is_file() and b.is_file())
    cleaned = origin != "raw with model-column cleaning"
    dtype = {"gnaf_id":"string", "mb":"string"}
    # Match the notebook's read call, dtypes and row sequence exactly.
    flats = pd.read_csv(a, dtype=dtype)[BASE].copy()
    ads = pd.read_csv(b, dtype=dtype, parse_dates=["listing_date"])[BASE+["listing_date","weekly_rent"]].copy()
    if not cleaned:
        # Recreate exactly the previously agreed cleaning for the columns this model uses.
        for df in (flats,ads):
            for c in ["gnaf_id","building_address","suburb","property_type"]:
                s = df[c].astype("string").str.strip().replace("",pd.NA)
                df[c] = s.str.lower() if c=="property_type" else s
            beds = df.bedrooms.astype(float).copy()
            baths = df.bathrooms.astype(float).copy()
            df["bedrooms"] = beds.mask(beds.notna() & (beds.mod(1).ne(0) | (beds.ge(10) & baths.le(2))))
            df["bathrooms"] = baths.mask(baths.ge(10) & beds.le(2))
    ads["listing_date"] = pd.to_datetime(ads.listing_date, format="%Y-%m-%d", errors="raise")
    assert len(flats)==65777 and not flats.gnaf_id.isna().any() and flats.gnaf_id.is_unique
    assert len(ads)==112457 and not ads.duplicated(KEY).any()
    assert ads.listing_date.notna().all() and ads.listing_date.max()==CUTOFF
    assert np.isfinite(ads.weekly_rent).all() and ads.weekly_rent.gt(0).all()
    for df in [flats,ads]:
        assert df.car_space.dropna().isin([0,1]).all()
    if raw_flats.exists():
        official = pd.read_csv(raw_flats, usecols=["gnaf_id"],dtype=dtype)
        assert flats.gnaf_id.equals(official.gnaf_id), "Official ID order differs"
    return flats,ads,{"source":origin,
                      "flats_sha256":digest(a),"ads_sha256":digest(b)}

def add_history(rows, past, origin=None):
    out = rows.reset_index(drop=True).copy()
    query = out[["gnaf_id", *BUILDING]].copy()
    query["asof"] = out.listing_date if origin is None else pd.Timestamp(origin)
    query["pos"] = np.arange(len(query))
    lookup = past[["gnaf_id", "listing_date", "weekly_rent"]].sort_values("listing_date")
    previous = pd.merge_asof(query.sort_values("asof"), lookup, left_on="asof",
        right_on="listing_date", by="gnaf_id", direction="backward",
        allow_exact_matches=False).sort_values("pos")
    out["prior_flat_rent"] = previous.weekly_rent.to_numpy()
    out["days_since_ad"] = (previous["asof"] - previous.listing_date).dt.days.to_numpy()

    # A comparable is a prior ad in the same suburb, building and bedroom group.
    # Left boundary inclusive, right boundary exclusive. No current-day/validation labels.
    p = past.sort_values("listing_date", kind="stable").reset_index(drop=True)
    groups = p.groupby(BUILDING, dropna=True, observed=True).indices
    dates = p.listing_date.to_numpy(dtype="datetime64[D]")
    rents = p.weekly_rent.to_numpy()
    origins = query["asof"].to_numpy(dtype="datetime64[D]")
    med, n = np.full(len(out), np.nan), np.zeros(len(out), dtype=int)
    for key, qidx in query.groupby(BUILDING, dropna=True, observed=True).indices.items():
        idx = groups.get(key)
        if idx is None:
            continue
        ds, ys, now = dates[idx], rents[idx], origins[qidx]
        lo = ds.searchsorted(now - np.timedelta64(365, "D"), side="left")
        hi = ds.searchsorted(now, side="left")
        n[qidx] = hi-lo
        for row, left, right in zip(qidx, lo, hi):
            if right-left >= 3:
                med[row] = np.median(ys[left:right])
    out["building_rent_365d"], out["building_n_365d"] = med, n
    return out

def fit_at(ads, origin):
    past = ads.loc[ads.listing_date < origin].sort_values("listing_date",kind="stable").reset_index(drop=True)
    start = origin - pd.DateOffset(months=10)
    tr = add_history(past.loc[past.listing_date >= start], past)
    X = tr[[*RAW_NUM,*FEATURES]].copy()
    X["market_days"] = (tr.listing_date - start).dt.days
    dtypes = {}
    for c in CATS:
        dtypes[c] = pd.CategoricalDtype(categories=sorted(tr[c].dropna().unique()))
        X[c] = tr[c].astype(dtypes[c])
    model = HistGradientBoostingRegressor(**PARAMS)
    model.fit(X,tr.weekly_rent)
    return model,past,start,dtypes,len(tr)

def query(rows, past, start, origin, dtypes):
    r = add_history(rows.drop(columns=["weekly_rent"],errors="ignore"),past,origin=origin)
    x = r[[*RAW_NUM,*FEATURES]].copy()
    x["market_days"] = (origin-start).days
    for c in CATS:
        x[c] = r[c].astype(dtypes[c])
    return x

def evaluate(ads, begin, end):
    origin = pd.Timestamp(begin)
    model,past,start,dtypes,nt = fit_at(ads,origin)
    window = ads.loc[ads.listing_date.between(origin,pd.Timestamp(end),inclusive="left")]
    va = window.loc[window.car_space.isin([0,1])].sort_values(KEY).reset_index(drop=True)
    pred = model.predict(query(va,past,start,origin,dtypes))
    assert np.isfinite(pred).all()
    err = np.abs(va.weekly_rent.to_numpy()-pred)
    return {"train_start":str(start.date()),"feature_origin":str(origin.date()),"train_n":nt,
            "window_n":len(window),"scored_n":len(va),"unknown_parking_excluded":len(window)-len(va),
            "MedAE_AUD_per_week":float(np.median(err)),"within_100_fraction":float(np.mean(err<=100)),
            "r2":float(r2_score(va.weekly_rent,pred)),
            "validation_keys_sha256":hashlib.sha256(va[KEY].to_csv(index=False,date_format="%Y-%m-%d",lineterminator="\n").encode()).hexdigest()}

def run(root, out):
    dst, report = out/"rents_vcts.csv",out/"reports"/"hgb_final"/"holdout_metrics.json"
    if dst.exists() or report.exists():
        raise SystemExit(f"Existing final file or report. Read the saved result; refusing to overwrite: {dst}, {report}")
    record_path = root/"reports"/"hgb_cv"/"hgb_selected_config.json"
    fold_path = root/"reports"/"hgb_cv"/"hgb_selected_cv.csv"
    chosen = json.loads(record_path.read_text(encoding="utf-8"))
    if chosen["params"] != PARAMS or chosen["use_history"] is not True or chosen["use_transit"] is not False or chosen["history_features"] != FEATURES:
        raise ValueError("与 notebook 已保存配置不一致，未评分或导出。")
    third = pd.read_csv(fold_path).loc[lambda x: x["折"].eq(3)]
    if len(third)!=1 or int(third.iloc[0]["n"])!=1917:
        raise ValueError("没有找到选定配置的第 3 折成绩。")
    ref = third.iloc[0]
    flats,ads,source=load(root)
    with threadpool_limits(limits=4):
        check = evaluate(ads,"2026-03-01","2026-05-01")
        if (check["validation_keys_sha256"] != HASH_MAR or check["train_n"] != 12387
                or not np.isclose(check["MedAE_AUD_per_week"],float(ref["median_absolute_error"]),atol=0.01,rtol=0)
                or not np.isclose(check["r2"],float(ref["r2"]),atol=1e-4,rtol=0)
                or not np.isclose(check["within_100_fraction"],float(ref["within_100_fraction"]),atol=1e-8,rtol=0)):
            raise RuntimeError(f"March–April notebook result not reproduced; holdout not accessed: {check}")
        print(f"Development reproduction: {check['scored_n']} ads, MedAE={check['MedAE_AUD_per_week']:.2f}",flush=True)
        # All features and settings above are fixed. This score does not choose or change the model.
        hold = evaluate(ads,"2026-05-01","2026-08-01")
        origin = CUTOFF+pd.Timedelta(days=1)
        model,past,start,dtypes,nt = fit_at(ads,origin)
        X = query(flats,past,start,origin,dtypes)
        sub = pd.DataFrame({"gnaf_id":flats.gnaf_id,
              "rent_with_car_space":model.predict(X.assign(car_space=1.0)).round(2),
              "rent_without_car_space":model.predict(X.assign(car_space=0.0)).round(2)})
    assert list(sub.columns)==SUB and len(sub)==65777 and sub.gnaf_id.equals(flats.gnaf_id)
    assert sub.gnaf_id.is_unique and sub.gnaf_id.notna().all()
    prices=sub[SUB[1:]].to_numpy()
    assert np.isfinite(prices).all() and (prices>0).all()
    dst.parent.mkdir(parents=True,exist_ok=True);report.parent.mkdir(parents=True,exist_ok=True)
    sub.to_csv(dst,index=False,mode="x",float_format="%.2f",lineterminator="\n")
    saved=pd.read_csv(dst,dtype={"gnaf_id":"string"})
    assert list(saved.columns)==SUB and saved.gnaf_id.equals(flats.gnaf_id)
    np.testing.assert_allclose(saved[SUB[1:]].to_numpy(),prices,rtol=0,atol=1e-8)
    metrics={"team":"vcts","model":"HGB + past asking rents, no transit","params":PARAMS,
        "source":source,"versions":{"python":sys.version.split()[0],"pandas":pd.__version__,
         "numpy":np.__version__,"scikit_learn":sklearn.__version__},
        "development_threefold_MedAE":chosen["three_fold_mean_MedAE"],
        "development_reference_fold":check,"holdout":hold,
        "final":{"info_cutoff":str(CUTOFF.date()),"train_start":str(start.date()),
         "train_n":nt,"available_prior_ads":len(past),"rows":len(sub),"columns":SUB,
         "pairs_with_lower_predicted_rent_with_parking":int((prices[:,0]<prices[:,1]).sum()),
         "csv_sha256":digest(dst),"csv_file":dst.name}}
    report.write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"本地留出集：{hold['scored_n']:,}/{hold['window_n']:,} 条 | MedAE=A${hold['MedAE_AUD_per_week']:.2f}/周 | A$100 内={hold['within_100_fraction']:.2%} | R²={hold['r2']:.4f}")
    print(f"最终训练：{nt:,} 条近期广告，另可使用 {len(past):,} 条截止日前历史记录。")
    print(f"提交文件：{dst}（{len(sub):,} 行，3 列；已重新读取核验）")
    print(f"留出报告：{report}")
    print(f"有车位预测低于无车位：{metrics['final']['pairs_with_lower_predicted_rent_with_parking']} 行（如实记录，未事后修改预测）。")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--output-dir", type=Path, default=Path("reproduced"))
    args = p.parse_args()
    run(args.root.resolve(), args.output_dir.resolve())
