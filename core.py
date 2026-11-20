"""FYP-Pulse core logic: cleaning (A1), features, model training (A3), tiers (A4),
explanations (A5), fairness (A5), watchlist (A6) and what-if (A7)."""
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)

SEED = 42

# ---------------------------------------------------------------- A1 cleaning
VALID_RANGES = {
    "group_size": (1, 5),
    "team_avg_cgpa": (0, 4),
    "team_min_cgpa": (0, 4),
    "supervisor_experience_years": (0, 45),
    "supervisor_active_groups": (1, 15),
    "proposal_revisions": (0, 10),
    "weeks_to_proposal_approval": (0, 30),
    "supervisor_meetings_8wk": (0, 30),
    "github_commits_8wk": (0, 1000),
    "mid_eval_score": (0, 100),
    "final_defense_score": (0, 100),
}
INTEGER_COLS = ["group_size", "supervisor_experience_years", "supervisor_active_groups",
                "proposal_revisions", "weeks_to_proposal_approval",
                "supervisor_meetings_8wk", "github_commits_8wk"]
CAMPUSES = ["Lahore", "Karachi", "Islamabad", "Faisalabad", "Multan", "Rawalpindi", "Peshawar", "Quetta"]
CAMPUS_ALIASES = {"lhr": "Lahore", "khi": "Karachi", "isb": "Islamabad", "fsd": "Faisalabad",
                  "rwp": "Rawalpindi", "pindi": "Rawalpindi", "pesh": "Peshawar"}
CATEGORICAL_VALID = {
    "program": ["BSCS", "BSSE", "BSAI", "BSIT", "BSDS"],
    "project_type": ["Web App", "Mobile App", "AI-ML", "IoT", "Research"],
    "has_industry_sponsor": ["Yes", "No"],
    "leader_gender": ["Male", "Female"],
    "extension_granted": ["Yes", "No"],
    "poor_outcome": ["Yes", "No"],
}


def _std_col(c):
    c = re.sub(r"[^0-9a-z]+", "_", str(c).strip().lower()).strip("_")
    return {"extension_grant": "extension_granted"}.get(c, c)


def _parse_date(s):
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%b-%y", "%d-%b-%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return pd.to_datetime(s, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT


def _std_campus(v):
    if pd.isna(v):
        return np.nan
    k = str(v).strip().lower()
    if k in CAMPUS_ALIASES:
        return CAMPUS_ALIASES[k]
    for c in CAMPUSES:
        if k == c.lower():
            return c
    return np.nan


def _std_gender(v):
    if pd.isna(v):
        return np.nan
    k = str(v).strip().lower()
    return {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}.get(k, np.nan)


def _std_choice(v, valid):
    if pd.isna(v):
        return np.nan
    k = str(v).strip().lower().replace("_", "-")
    for x in valid:
        if k == x.lower() or k.replace(" ", "") == x.lower().replace(" ", ""):
            return x
    return np.nan


def clean(raw: pd.DataFrame):
    """Returns (clean_df, rule_report_df, column_summary_df, summary_dict)."""
    df = raw.copy()
    df.columns = [_std_col(c) for c in df.columns]
    rules = []
    per_col = {c: 0 for c in df.columns}
    initial_rows = len(df)

    # 1 duplicates (exact, then repeated group_id)
    n_exact = int(df.duplicated().sum())
    df = df.drop_duplicates()
    n_id = int(df["group_id"].duplicated().sum())
    df = df.drop_duplicates(subset="group_id", keep="first").reset_index(drop=True)
    rules.append(("Duplicate rows", n_exact + n_id,
                  f"Removed {n_exact} exact duplicates and {n_id} repeated group IDs (kept first)."))

    # 2 text standardisation
    before = df["campus"].copy()
    df["campus"] = df["campus"].map(_std_campus)
    n = int((before.astype(str) != df["campus"].astype(str)).sum())
    per_col["campus"] += n
    rules.append(("Inconsistent campus spelling", n,
                  "Mapped case variants, trailing spaces and short forms (Lhr, KHI, ISB, Fsd, RWP, Pindi, Pesh) to the 8 official names."))

    before = df["leader_gender"].copy()
    df["leader_gender"] = df["leader_gender"].map(_std_gender)
    n = int((before.astype(str) != df["leader_gender"].astype(str)).sum())
    per_col["leader_gender"] += n
    rules.append(("Gender coded as M / F", n, "Converted M to Male and F to Female."))

    for col, valid in CATEGORICAL_VALID.items():
        if col in ("leader_gender",) or col not in df:
            continue
        before = df[col].copy()
        df[col] = df[col].map(lambda v: _std_choice(v, valid))
        n = int((before.notna() & (before.astype(str) != df[col].astype(str))).sum())
        if n:
            per_col[col] += n
            rules.append((f"Invalid {col} label", n, "Mapped to a valid label or set to missing."))

    # 3 dates
    parsed = df["proposal_date"].map(_parse_date)
    iso = df["proposal_date"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")
    n_fmt = int((~iso & parsed.notna()).sum())
    n_bad = int(parsed.isna().sum())
    df["proposal_date"] = parsed.dt.strftime("%Y-%m-%d")
    per_col["proposal_date"] += n_fmt + n_bad
    rules.append(("Mixed date formats", n_fmt,
                  "Converted DD/MM/YYYY and DD-Mon-YY dates to YYYY-MM-DD."
                  + (f" {n_bad} unparseable dates set to missing." if n_bad else "")))

    # 4 impossible values -> missing (filled in step 5)
    oor = {}
    for col, (lo, hi) in VALID_RANGES.items():
        bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        if col in INTEGER_COLS:
            bad |= df[col].notna() & (df[col] % 1 != 0)
        n = int(bad.sum())
        if n:
            ex = ", ".join(str(v).rstrip("0").rstrip(".") for v in sorted(df.loc[bad, col].unique())[:4])
            df.loc[bad, col] = np.nan
            per_col[col] += n
            oor[col] = n
            rules.append((f"Out of range: {col}", n,
                          f"Values outside {lo}-{hi} (e.g. {ex}) treated as errors, set to missing, then filled (see the Missing rules below)."))

    # 5 missing values. CGPA first, using the other CGPA so min <= avg stays true
    orig_bad = int((df["team_min_cgpa"] > df["team_avg_cgpa"]).sum())
    gap = (df["team_avg_cgpa"] - df["team_min_cgpa"]).clip(lower=0).median()
    for col, other, sign in (("team_avg_cgpa", "team_min_cgpa", 1), ("team_min_cgpa", "team_avg_cgpa", -1)):
        miss = df[col].isna()
        n_orig = int(miss.sum()) - oor.get(col, 0)
        use_other = miss & df[other].notna()
        df.loc[use_other, col] = (df.loc[use_other, other] + sign * gap).clip(0, 4)
        df.loc[df[col].isna(), col] = df[col].median()
        per_col[col] += n_orig
        rules.append((f"Missing: {col}", n_orig,
                      f"Filled from the team's other CGPA (+/- median gap of {gap:.2f}); median if both missing."))
    for col in df.columns:
        if col in ("group_id",):
            continue
        n = int(df[col].isna().sum())
        if not n:
            continue
        if col in VALID_RANGES:
            fill = df[col].median()
            if col in INTEGER_COLS:
                fill = round(fill)
            how = f"median ({fill:g})"
        else:
            fill = df[col].mode().iloc[0]
            how = f"most common value ({fill})"
        df[col] = df[col].fillna(fill)
        n_orig = n - oor.get(col, 0)
        per_col[col] += n_orig
        if n_orig:
            rules.append((f"Missing: {col}", n_orig, f"Filled with {how}."))

    # 6 logical CGPA rules
    bad = df["team_min_cgpa"] > df["team_avg_cgpa"]
    n = int(bad.sum())
    df.loc[bad, "team_min_cgpa"] = df.loc[bad, "team_avg_cgpa"]
    per_col["team_min_cgpa"] += n
    rules.append(("team_min_cgpa above team_avg_cgpa", n,
                  "The weakest member cannot be above the average, so min was capped at the average."))
    solo = (df["group_size"] == 1) & (df["team_min_cgpa"] != df["team_avg_cgpa"])
    n = int(solo.sum())
    df.loc[solo, "team_min_cgpa"] = df.loc[solo, "team_avg_cgpa"]
    per_col["team_min_cgpa"] += n
    rules.append(("Solo group with min != avg", n,
                  "In a one-person group min and avg must match, so min was set to the average."))

    for col in INTEGER_COLS:
        df[col] = df[col].round().astype(int)
    df["poor_outcome_flag"] = (df["poor_outcome"] == "Yes").astype(int)

    rule_df = pd.DataFrame(rules, columns=["Rule broken", "Rows affected", "How it was fixed"])
    col_df = (pd.Series(per_col, name="Values corrected").rename_axis("Column").reset_index())
    col_df = col_df[col_df["Values corrected"] > 0].sort_values("Values corrected", ascending=False)
    summary = {
        "initial_rows": initial_rows,
        "clean_rows": len(df),
        "values_corrected": int(col_df["Values corrected"].sum()),
        "poor_outcomes": int(df["poor_outcome_flag"].sum()),
        "poor_rate": float(df["poor_outcome_flag"].mean()),
    }
    return df, rule_df, col_df.reset_index(drop=True), summary


# ---------------------------------------------------------------- features
RAW_NUM = ["group_size", "team_avg_cgpa", "team_min_cgpa", "supervisor_experience_years",
           "supervisor_active_groups", "proposal_revisions", "weeks_to_proposal_approval",
           "supervisor_meetings_8wk", "github_commits_8wk", "mid_eval_score"]
RAW_CAT = ["program", "project_type", "has_industry_sponsor"]
ENGINEERED = {
    "cgpa_gap": "Team average CGPA minus the weakest member's CGPA. A big gap means one member is "
                "far behind the rest, so work piles onto the others and that member can stall the project.",
    "commits_per_member": "GitHub commits in the first 8 weeks divided by group size. Raw commits "
                          "favour big teams; per-member activity shows how much each student is really building.",
    "meetings_per_load": "Supervisor meetings in 8 weeks divided by the supervisor's active groups. "
                         "Shows how much attention this group gets relative to how stretched the supervisor is.",
}
FEATURES_NUM = RAW_NUM + list(ENGINEERED)
INPUT_FIELDS = RAW_NUM + RAW_CAT          # what a user enters at week 8

DROPPED = [
    ("final_defense_score", "Leakage", "Only known at the end of the project. It is basically the answer."),
    ("extension_granted", "Leakage", "Decided in the final phase, long after week 8. 74% of groups with an extension ended badly, so it would fake accuracy."),
    ("group_id", "Identifier", "A random label with no meaning about the group."),
    ("proposal_date", "Not predictive / cohort", "A calendar date ties the model to past intakes rather than to the group's situation."),
    ("leader_gender", "Fairness", "A protected attribute. Kept out of the model and used only for the fairness check (A5)."),
    ("campus", "Fairness", "Location is not something support can act on and shows little signal after cleaning. Used only for the fairness check (A5)."),
    ("poor_outcome", "Target", "This is what we predict."),
]

FRIENDLY = {
    "group_size": "Group size", "team_avg_cgpa": "Team average CGPA", "team_min_cgpa": "Weakest member's CGPA",
    "supervisor_experience_years": "Supervisor experience (years)", "supervisor_active_groups": "Supervisor's active groups",
    "proposal_revisions": "Proposal revisions", "weeks_to_proposal_approval": "Weeks to proposal approval",
    "supervisor_meetings_8wk": "Supervisor meetings (8 wk)", "github_commits_8wk": "GitHub commits (8 wk)",
    "mid_eval_score": "Mid-evaluation score", "program": "Program", "project_type": "Project type",
    "has_industry_sponsor": "Industry sponsor", "cgpa_gap": "CGPA gap (avg - min)",
    "commits_per_member": "Commits per member", "meetings_per_load": "Meetings per supervisor load",
}


def add_features(df):
    df = df.copy()
    df["cgpa_gap"] = (df["team_avg_cgpa"] - df["team_min_cgpa"]).clip(lower=0)
    df["commits_per_member"] = df["github_commits_8wk"] / df["group_size"].clip(lower=1)
    df["meetings_per_load"] = df["supervisor_meetings_8wk"] / df["supervisor_active_groups"].clip(lower=1)
    return df


def _pre():
    return ColumnTransformer([
        ("num", StandardScaler(), FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), RAW_CAT),
    ])


def model_zoo():
    return {
        "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=2000, C=0.5),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced_subsample",
                                                random_state=SEED, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=5, class_weight="balanced",
                                            random_state=SEED, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=250, max_leaf_nodes=15,
                                                            l2_regularization=1.0, class_weight="balanced",
                                                            random_state=SEED),
    }


def _best_f1_threshold(y, p):
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def _metrics(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0), "F1": f1_score(y, pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y, p), "PR-AUC": average_precision_score(y, p),
            "Threshold": t, "cm": [[int(tn), int(fp)], [int(fn), int(tp)]]}


def train_all(clean_df):
    df = add_features(clean_df)
    X = df[FEATURES_NUM + RAW_CAT]
    y = df["poor_outcome_flag"].values
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=SEED)
    Xtr, Xte, ytr, yte = X.iloc[idx_tr], X.iloc[idx_te], y[idx_tr], y[idx_te]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results, fitted, oof_store = [], {}, {}
    for name, est in model_zoo().items():
        pipe = Pipeline([("pre", _pre()), ("clf", est)])
        oof = cross_val_predict(pipe, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        t = _best_f1_threshold(ytr, oof)          # threshold chosen on CV, never on test
        pipe.fit(Xtr, ytr)
        pte = pipe.predict_proba(Xte)[:, 1]
        m = _metrics(yte, pte, t)
        m.update({"Model": name, "CV ROC-AUC": roc_auc_score(ytr, oof),
                  "CV PR-AUC": average_precision_score(ytr, oof), "CV F1": f1_score(ytr, (oof >= t).astype(int))})
        results.append(m)
        fitted[name] = pipe
        oof_store[name] = oof

    res = pd.DataFrame(results)
    best = res.sort_values(["CV PR-AUC", "CV ROC-AUC"], ascending=False).iloc[0]["Model"]
    model = fitted[best]
    threshold = float(res.set_index("Model").loc[best, "Threshold"])

    # tiers from CV probabilities of the chosen model
    oof = oof_store[best]
    t_high = threshold
    cands = np.round(np.arange(0.02, t_high, 0.01), 2)
    t_low = float(cands[0])
    for c in cands:  # largest cut-off where Low tier's poor-outcome rate stays <= 5%
        mask = oof < c
        if mask.sum() >= 30 and ytr[mask].mean() <= 0.05:
            t_low = float(c)
    t_low = min(t_low, t_high - 0.05)

    test = df.iloc[idx_te].copy()
    test["probability"] = model.predict_proba(Xte)[:, 1]
    test["predicted_poor"] = (test["probability"] >= threshold).astype(int)
    test["tier"] = test["probability"].map(lambda p: tier_of(p, t_low, t_high))
    train = df.iloc[idx_tr].copy()
    reference = {c: float(train[c].median()) for c in RAW_NUM}
    reference.update({c: train[c].mode().iloc[0] for c in RAW_CAT})
    return {"results": res, "best": best, "model": model, "threshold": threshold,
            "t_low": t_low, "t_high": t_high, "test": test, "train": train,
            "reference": reference, "fitted": fitted}


# ---------------------------------------------------------------- A4/A5 prediction & explanation
def tier_of(p, t_low, t_high):
    return "High" if p >= t_high else ("Medium" if p >= t_low else "Low")


def clean_input(row: dict):
    """Enforce the A1 CGPA rules on a single entered group: min <= avg, and solo group min == avg."""
    row = dict(row)
    row["team_min_cgpa"] = min(row["team_min_cgpa"], row["team_avg_cgpa"])
    if row["group_size"] == 1:
        row["team_min_cgpa"] = row["team_avg_cgpa"]
    return row


def predict_one(model, row: dict):
    X = add_features(pd.DataFrame([clean_input(row)]))[FEATURES_NUM + RAW_CAT]
    return float(model.predict_proba(X)[:, 1][0])


def explain(model, row: dict, reference: dict):
    """Change each input to the typical (training median/mode) value and measure how much the risk moves.
    Positive impact = this input pushes risk UP versus a typical group."""
    row = clean_input(row)
    base = predict_one(model, row)
    rows = []
    for f in INPUT_FIELDS:
        alt = dict(row)
        alt[f] = reference[f]
        if f == "team_avg_cgpa":  # move the whole team's CGPA, keep the avg-to-min gap fixed
            alt["team_min_cgpa"] = max(0.0, reference[f] - (row["team_avg_cgpa"] - row["team_min_cgpa"]))
        impact = base - predict_one(model, alt)
        rows.append({"feature": f, "label": FRIENDLY[f], "value": row[f], "typical": reference[f], "impact": impact})
    ex = pd.DataFrame(rows)
    return base, ex.sort_values("impact", ascending=False).reset_index(drop=True)


def fmt_val(f, v):
    if isinstance(v, str):
        return v
    if "cgpa" in f:
        return f"{v:.2f}"
    if f == "mid_eval_score":
        return f"{v:.1f}"
    return f"{v:.0f}"


SUPPORT = {
    "supervisor_meetings_8wk": "Few supervisor meetings: book a fixed weekly 30-minute slot with the supervisor.",
    "github_commits_8wk": "Low coding activity: set weekly code milestones and a short Friday demo.",
    "mid_eval_score": "Weak mid-evaluation: hold a follow-up review within 2 weeks to fix the gaps the panel flagged.",
    "team_min_cgpa": "Weakest member is struggling: pair them with a stronger teammate and give them one clearly owned module.",
    "team_avg_cgpa": "Academically weaker team: offer a lab mentor or TA check-in every two weeks.",
    "supervisor_active_groups": "Supervisor is overloaded: add a co-supervisor or move the group to a lighter load.",
    "supervisor_experience_years": "New supervisor: pair them with an experienced co-supervisor for this group.",
    "proposal_revisions": "Scope kept changing at proposal: run a scope-cutting session to lock a minimum viable version.",
    "weeks_to_proposal_approval": "Slow start: agree a catch-up plan with dated milestones for the next 4 weeks.",
    "group_size": "Team size is a risk: rebalance work with one named owner per module and a shared task board.",
    "project_type": "Higher-risk project type: set fortnightly checkpoints with concrete deliverables.",
    "has_industry_sponsor": "No industry sponsor: connect the group with an industry or alumni mentor.",
    "program": "Program-level risk: invite the group to the department's FYP clinic sessions.",
}


def narrative(prob, tier, ex):
    up = ex[ex["impact"] > 0.005].head(3)
    down = ex[ex["impact"] < -0.005].sort_values("impact").head(3)
    s = f"This group has a **{prob:.0%} estimated chance** of failing or carrying over, which puts it in the **{tier} risk** tier. "
    if len(up):
        bits = [f"{r.label.lower()} of {fmt_val(r.feature, r.value)} (typical {fmt_val(r.feature, r.typical)})" for r in up.itertuples()]
        s += "The main things pushing risk up are " + "; ".join(bits) + ". "
    if len(down):
        bits = [f"{r.label.lower()} of {fmt_val(r.feature, r.value)}" for r in down.itertuples()]
        s += "Working in its favour: " + "; ".join(bits) + "."
    action = SUPPORT[up.iloc[0]["feature"]] if len(up) else "No single factor stands out. Keep the normal check-in schedule."
    return s, action


# ---------------------------------------------------------------- A5 fairness
def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return (c - h, c + h)


def group_table(test, col, min_n=0):
    rows = []
    for g, d in test.groupby(col):
        if len(d) < min_n:
            continue
        pos = d[d["poor_outcome_flag"] == 1]
        tp = int(pos["predicted_poor"].sum())
        lo, hi = wilson(tp, len(pos))
        rows.append({col: g, "Test groups": len(d), "Actual poor outcomes": len(pos),
                     "Actual poor rate": d["poor_outcome_flag"].mean(),
                     "Predicted poor rate": d["predicted_poor"].mean(),
                     "Recall": tp / len(pos) if len(pos) else np.nan,
                     "Recall 95% CI": f"{lo:.0%} - {hi:.0%}" if len(pos) else "n/a"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- A7 levers
LEVERS = {
    "supervisor_meetings_8wk": ("Supervisor meetings (8 wk)", 0, 30, 1, lambda v: max(v, 8), "weekly meetings (8)"),
    "github_commits_8wk": ("GitHub commits (8 wk)", 0, 1000, 5, lambda v: max(v, 90), "steady coding (90+ commits)"),
    "supervisor_active_groups": ("Supervisor's active groups", 1, 15, 1, lambda v: min(v, 4), "lighter supervisor load (4 groups)"),
    "supervisor_experience_years": ("Supervisor experience (years)", 0, 45, 1, lambda v: max(v, 10), "experienced co-supervisor (10 yrs)"),
    "has_industry_sponsor": ("Industry sponsor", None, None, None, lambda v: "Yes", "industry sponsor or mentor"),
}


def best_lever(model, row: dict):
    """Apply each lever's realistic target on its own; return a table sorted by risk reduction (pts)."""
    base = predict_one(model, row)
    out = []
    for f, (label, *_rest, target, desc) in LEVERS.items():
        alt = dict(row)
        alt[f] = target(row[f])
        p = predict_one(model, alt)
        out.append({"Lever": label, "Change": f"{fmt_val(f, row[f])} -> {fmt_val(f, alt[f])} ({desc})",
                    "New probability": p, "Change (pts)": (p - base) * 100})
    return pd.DataFrame(out).sort_values("Change (pts)").reset_index(drop=True)


def watchlist(test, model, reference, k):
    top = test.sort_values("probability", ascending=False).head(k).copy()
    drivers = []
    for _, r in top.iterrows():
        _, ex = explain(model, {f: r[f] for f in INPUT_FIELDS}, reference)
        drivers.append(ex.iloc[0]["label"] if ex.iloc[0]["impact"] > 0 else "none")
    top["top_risk_driver"] = drivers
    return top
