import pandas as pd
import streamlit as st
import core

st.set_page_config(page_title="FYP-Pulse", page_icon="📊", layout="wide")

st.markdown("""<style>
h1 {background: linear-gradient(90deg,#4F46E5,#DB2777); -webkit-background-clip:text; -webkit-text-fill-color:transparent;}
[data-testid="stMetric"] {background:#FFFFFF; border:1px solid #E0E7FF; border-left:5px solid #4F46E5; border-radius:12px; padding:12px 16px; box-shadow:0 1px 3px rgba(79,70,229,.08);}
.stTabs [data-baseweb="tab-list"] {gap:6px;}
.stTabs [data-baseweb="tab"] {background:#EEF2FF; border-radius:10px 10px 0 0; padding:8px 14px; font-weight:600;}
.stTabs [aria-selected="true"] {background:#4F46E5 !important; color:#FFFFFF !important;}
h2, h3 {color:#3730A3;}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    raw = pd.read_csv("fyp_pulse_train.csv")
    return (raw,) + core.clean(raw)


raw, df, rules, cols, s = load()

st.title("FYP-Pulse: Final Year Project Early Warning System")
st.caption("Uses only what is known at the end of week 8 to flag groups likely to fail or carry over.")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["A1 · Data Quality", "A2 · Dashboard", "A3 · Models", "A4 · Predict",
     "A5 · Explain & Fairness", "A6 · Watchlist", "A7 · What-if"])

# ------------------------------------------------------------------ A1
with tab1:
    st.header("A1 · Data Cleaning & Summary")
    c = st.columns(5)
    c[0].metric("Initial rows", f"{s['initial_rows']:,}")
    c[1].metric("Cleaned rows", f"{s['clean_rows']:,}", f"-{s['initial_rows'] - s['clean_rows']} duplicates")
    c[2].metric("Values corrected", f"{s['values_corrected']:,}")
    c[3].metric("Poor outcomes", f"{s['poor_outcomes']:,}")
    c[4].metric("Poor-outcome rate", f"{s['poor_rate']:.1%}")

    st.info(f"About 1 in {round(1 / s['poor_rate'])} groups ends badly. A model that says \"No risk\" for everyone "
            f"would look {1 - s['poor_rate']:.0%} accurate and catch nobody, so accuracy alone is not the measure here.")

    st.subheader("Rules broken and how each was fixed")
    st.dataframe(rules, hide_index=True, width="stretch")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Values corrected per column")
        st.dataframe(cols, hide_index=True, width="stretch")
    with right:
        st.subheader("Valid ranges enforced")
        st.dataframe(pd.DataFrame([(k, lo, hi) for k, (lo, hi) in core.VALID_RANGES.items()],
                                  columns=["Column", "Min", "Max"]), hide_index=True, width="stretch")
        st.caption("Any value outside its range is impossible, so it is treated as an error: set to missing, then filled.")

    with st.expander("Before vs after: sample rows"):
        st.write("Raw data")
        st.dataframe(raw.head(10), width="stretch")
        st.write("Cleaned data")
        st.dataframe(df.drop(columns="poor_outcome_flag").head(10), width="stretch")

    st.download_button("Download cleaned CSV", df.drop(columns="poor_outcome_flag").to_csv(index=False),
                       "fyp_pulse_clean.csv", "text/csv")

# ------------------------------------------------------------------ A2
import plotly.express as px

feat = core.add_features(df)
BASE = feat["poor_outcome_flag"].mean()


def rate_by(col, bins=None, labels=None, order=None):
    g = pd.cut(feat[col], bins=bins, labels=labels, include_lowest=True) if bins is not None else feat[col]
    out = (feat.groupby(g, observed=True)["poor_outcome_flag"].agg(rate="mean", groups="size")
           .reset_index().rename(columns={col: "band"}))
    out["band"] = out["band"].astype(str)
    if order:
        out = out.set_index("band").loc[order].reset_index()
    return out


def rate_chart(d, title, xlabel):
    d = d.copy()
    d["label"] = d.apply(lambda r: f"{r.rate:.0%}<br><span style='font-size:10px'>n={r.groups}</span>", axis=1)
    d["vs average"] = d["rate"].map(lambda r: "Above average risk" if r > BASE else "At or below average")
    fig = px.bar(d, x="band", y="rate", text="label", color="vs average",
                 color_discrete_map={"Above average risk": "#d9534f", "At or below average": "#5b8def"},
                 labels={"band": xlabel, "rate": "Poor-outcome rate"}, title=title)
    fig.add_hline(y=BASE, line_dash="dash", line_color="grey",
                  annotation_text=f"Overall {BASE:.0%}", annotation_position="top right", annotation_bgcolor="rgba(255,255,255,0.85)")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, max(d["rate"].max() * 1.3, BASE * 1.5)],
                      legend_title_text="", legend_orientation="h", legend_y=-0.25, height=420,
                      margin=dict(t=60, b=10))
    return fig


def insight(text):
    st.markdown(f"**Insight:** {text}")


with tab2:
    st.header("A2 · Dashboards & Visualizations")
    st.caption("Each bar is the share of groups in that band that failed or carried over. "
               f"Red bars are above the overall rate of {BASE:.0%}. n = number of groups.")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.markdown("##### 1 · Team factor: weakest member's CGPA")
        d = rate_by("team_min_cgpa", [0, 2, 2.5, 3, 3.5, 4], ["< 2.0", "2.0-2.5", "2.5-3.0", "3.0-3.5", "3.5+"])
        st.plotly_chart(rate_chart(d, "Poor-outcome rate by weakest member's CGPA", "Weakest member's CGPA"), width="stretch")
        lo, hi = d.iloc[0], d.iloc[-1]
        insight(f"A team is only as strong as its weakest member: groups whose weakest student is below 2.0 CGPA end badly "
                f"**{lo.rate:.0%}** of the time, against **{hi.rate:.0%}** when everyone is at 3.5+.")
    with r1c2:
        st.markdown("##### 2 · Supervisor factor: meetings in the first 8 weeks")
        d = rate_by("supervisor_meetings_8wk", [0, 2, 5, 8, 30], ["0-2", "3-5", "6-8", "9+"])
        st.plotly_chart(rate_chart(d, "Poor-outcome rate by supervisor meetings (8 wk)", "Meetings with supervisor"), width="stretch")
        insight(f"Groups that met their supervisor 0-2 times fail **{d.iloc[0].rate:.0%}** of the time; "
                f"roughly weekly contact (9+) cuts that to **{d.iloc[-1].rate:.0%}**, so early meetings are a cheap lever.")

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.markdown("##### 3 · Project context: project type and industry sponsor")
        d = (feat.groupby(["project_type", "has_industry_sponsor"])["poor_outcome_flag"]
             .agg(rate="mean", groups="size").reset_index())
        order = feat.groupby("project_type")["poor_outcome_flag"].mean().sort_values(ascending=False).index.tolist()
        fig = px.bar(d, x="project_type", y="rate", color="has_industry_sponsor", barmode="group",
                     category_orders={"project_type": order, "has_industry_sponsor": ["No", "Yes"]},
                     color_discrete_map={"No": "#d9534f", "Yes": "#5cb85c"},
                     text=d["rate"].map(lambda r: f"{r:.0%}"),
                     labels={"project_type": "Project type", "rate": "Poor-outcome rate",
                             "has_industry_sponsor": "Industry sponsor"},
                     title="Poor-outcome rate by project type and sponsor")
        fig.add_hline(y=BASE, line_dash="dash", line_color="grey", annotation_text=f"Overall {BASE:.0%}",
                      annotation_position="top right", annotation_bgcolor="rgba(255,255,255,0.85)")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(yaxis_tickformat=".0%", height=420, legend_orientation="h", legend_y=-0.25, margin=dict(t=60, b=10))
        st.plotly_chart(fig, width="stretch")
        pt = feat.groupby("project_type")["poor_outcome_flag"].mean()
        sp = feat.groupby("has_industry_sponsor")["poor_outcome_flag"].mean()
        insight(f"**{pt.idxmax()}** projects are the riskiest ({pt.max():.0%} vs {pt.min():.0%} for {pt.idxmin()}), "
                f"and groups with an industry sponsor fail less often ({sp['Yes']:.0%} vs {sp['No']:.0%}).")
    with r2c2:
        st.markdown("##### 4 · Engineered feature: commits per member")
        q = pd.qcut(feat["commits_per_member"], 5)
        labels = [f"{iv.left:.0f}-{iv.right:.0f}" for iv in q.cat.categories]
        labels[0] = f"0-{q.cat.categories[0].right:.0f}"
        d = rate_by("commits_per_member", list(q.cat.categories.left[:1]) + list(q.cat.categories.right), labels)
        st.plotly_chart(rate_chart(d, "Poor-outcome rate by commits per member (8 wk)", "GitHub commits per member (5 equal-sized groups)"),
                        width="stretch")
        insight(f"The least active fifth of groups (≤{q.cat.categories[0].right:.0f} commits per student) fail "
                f"**{d.iloc[0].rate:.0%}** of the time, while the most active fifth fail only **{d.iloc[-1].rate:.0%}**. "
                "Per-person effort says much more than the raw commit count.")

    st.markdown("##### What the engineered features measure")
    for k, v in core.ENGINEERED.items():
        st.markdown(f"- **{core.FRIENDLY[k]}**: {v}")
    st.caption("Commits per member is charted above. All three engineered features go into the model (A3).")

    with st.expander("Extra: the week-8 mid-evaluation"):
        d = rate_by("mid_eval_score", [0, 50, 60, 70, 80, 100], ["< 50", "50-60", "60-70", "70-80", "80+"])
        st.plotly_chart(rate_chart(d, "Poor-outcome rate by mid-evaluation score", "Mid-evaluation score"), width="stretch")
        insight(f"The mid-evaluation is the strongest single week-8 signal: below 50 means **{d.iloc[0].rate:.0%}** end badly, "
                f"80+ means only **{d.iloc[-1].rate:.0%}**.")


# ------------------------------------------------------------------ A3
import plotly.graph_objects as go
from sklearn.metrics import roc_curve


@st.cache_resource
def train(_df):
    return core.train_all(_df)


R = train(df)
test = R["test"]
Xte = test[core.FEATURES_NUM + core.RAW_CAT]
yte = test["poor_outcome_flag"].values

with tab3:
    st.header("A3 · Model Training & Selection")

    st.subheader("1 · Columns dropped and why")
    st.dataframe(pd.DataFrame(core.DROPPED, columns=["Column", "Reason", "Explanation"]), hide_index=True, width="stretch")
    st.markdown("**Features used (all known at week 8):** " +
                ", ".join(core.FRIENDLY[f] for f in core.RAW_NUM + core.RAW_CAT) +
                ", plus engineered: " + ", ".join(core.FRIENDLY[f] for f in core.ENGINEERED) + ".")

    st.subheader("2 · How the models were trained")
    n_tr, n_te = len(R["train"]), len(test)
    st.markdown(f"""
- **Stratified 80/20 split:** {n_tr} training groups, {n_te} test groups. Both keep the same ~{BASE:.0%} poor-outcome share.
- **5-fold stratified cross-validation** on the training set to compare models and to **choose each model's threshold** (the cut-off that maximises F1). The test set is never used to tune anything.
- **Class imbalance:** every model uses balanced class weights, so a missed struggling group costs more than a false alarm.
- **Preprocessing:** numeric features scaled, categorical features one-hot encoded, inside one pipeline (no leakage between folds).
""")

    st.subheader("3 · Test-set results")
    res = R["results"].copy()
    show = res[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Threshold", "CV ROC-AUC", "CV PR-AUC"]]
    st.dataframe(show.style.format({c: "{:.3f}" for c in show.columns if c != "Model"})
                 .highlight_max(subset=["F1", "ROC-AUC", "PR-AUC", "CV PR-AUC"], color="#d4edda"),
                 hide_index=True, width="stretch")
    st.caption("PR-AUC (average precision) is the fairest score when only 16% of groups are positive. "
               "Random guessing would score about 0.16.")

    st.markdown("**Confusion matrices on the test set** (at each model's threshold)")
    cms = st.columns(len(res))
    for col, r in zip(cms, res.itertuples()):
        cm = r.cm
        fig = go.Figure(go.Heatmap(z=cm, x=["Pred: OK", "Pred: Poor"], y=["Actual: OK", "Actual: Poor"],
                                   text=[[str(v) for v in row] for row in cm], texttemplate="%{text}",
                                   textfont={"size": 16}, colorscale="Blues", showscale=False))
        fig.update_layout(title=f"{r.Model}<br><sup>threshold {r.Threshold:.2f}</sup>", height=280,
                          margin=dict(t=60, b=10, l=10, r=10), yaxis_autorange="reversed")
        col.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    for name, pipe in R["fitted"].items():
        fpr, tpr, _ = roc_curve(yte, pipe.predict_proba(Xte)[:, 1])
        auc = res.set_index("Model").loc[name, "ROC-AUC"]
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {auc:.3f})",
                                 line=dict(width=4 if name == R["best"] else 2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="grey")))
    fig.update_layout(title="ROC curves on the test set", xaxis_title="False positive rate",
                      yaxis_title="True positive rate (recall)", height=420)
    st.plotly_chart(fig, width="stretch")

    st.subheader("4 · Final model and why")
    b = res.set_index("Model").loc[R["best"]]
    st.success(f"**Selected: {R['best']}** with a decision threshold of **{R['threshold']:.2f}**.")
    st.markdown(f"""
- **Best on cross-validation:** highest CV PR-AUC ({b['CV PR-AUC']:.3f}) and CV ROC-AUC ({b['CV ROC-AUC']:.3f}). It was chosen on the training folds, not by peeking at the test set.
- **Holds up on the test set:** ROC-AUC {b['ROC-AUC']:.3f} and F1 {b['F1']:.3f}. It catches **{b['Recall']:.0%}** of struggling groups, and **{b['Precision']:.0%}** of the groups it flags really end badly (vs {BASE:.0%} if picked at random).
- **Explainable:** each feature has one clear direction, so the coordinator can see *why* a group is flagged. That matters for an advisory tool used on students.
- **Simple and stable:** no tree model beat it on cross-validation. Gradient Boosting is slightly higher on test PR-AUC, but it only gets there by flagging far more groups (precision {res.set_index('Model').loc['Gradient Boosting', 'Precision']:.0%}), with lower F1 and accuracy. With ~1,400 training rows, extra complexity adds variance, not signal.
""")


# ------------------------------------------------------------------ shared for A4-A7
MODEL, T_LOW, T_HIGH, THRESHOLD, REF = R["model"], R["t_low"], R["t_high"], R["threshold"], R["reference"]
TIER_COLOR = {"Low": "#2e7d32", "Medium": "#f0ad00", "High": "#c62828"}
TIERS = ["Low", "Medium", "High"]
ranked = test.sort_values("probability", ascending=False)
med = ranked[ranked["tier"] == "Medium"]
EXAMPLES = {"High": ranked.iloc[0]["group_id"], "Medium": med.iloc[len(med) // 2]["group_id"],
            "Low": ranked.iloc[-1]["group_id"]}
OPTS = {"program": core.CATEGORICAL_VALID["program"], "project_type": core.CATEGORICAL_VALID["project_type"],
        "has_industry_sponsor": ["No", "Yes"]}


def set_form(r):
    for f in core.INPUT_FIELDS:
        v = r[f]
        if f in core.INTEGER_COLS:
            v = int(v)
        elif f not in core.RAW_CAT:
            v = float(v)
        st.session_state[f"in_{f}"] = v


if "in_group_size" not in st.session_state:
    set_form(test.loc[test.group_id == EXAMPLES["Medium"]].iloc[0])


def load_example():
    set_form(test.loc[test.group_id == st.session_state["ex_gid"]].iloc[0])
    for f in core.LEVERS:
        st.session_state.pop(f"lv_{f}", None)


def badge(tier, p):
    c = TIER_COLOR[tier]
    return (f'<span style="background:{c};color:white;border-radius:999px;padding:6px 18px;font-size:20px;'
            f'font-weight:700">● {tier} risk</span>&nbsp;&nbsp;<span style="font-size:26px;font-weight:700">{p:.1%}</span>'
            f'&nbsp;<span style="opacity:.7">probability of a poor outcome</span>')


# ------------------------------------------------------------------ A4
with tab4:
    st.header("A4 · Prediction Form")
    with st.container(border=True):
        e1, e2, e3 = st.columns([1, 2, 1])
        tier_pick = e1.radio("Load an example test group", TIERS[::-1], horizontal=True, key="ex_tier")
        ids = ranked[ranked["tier"] == tier_pick]["group_id"].tolist()
        e2.selectbox(f"Test group ({len(ids)} in tier)", ids, index=ids.index(EXAMPLES[tier_pick]), key="ex_gid")
        e3.button("Load into form", on_click=load_example, type="primary")
    with st.form("predict"):
        c4 = st.columns(3)
        for i, f in enumerate(core.INPUT_FIELDS):
            box, label = c4[i % 3], core.FRIENDLY[f]
            if f in core.RAW_CAT:
                box.selectbox(label, OPTS[f], key=f"in_{f}")
            else:
                lo, hi = core.VALID_RANGES[f]
                if f in core.INTEGER_COLS:
                    box.number_input(label, int(lo), int(hi), step=1, key=f"in_{f}")
                else:
                    box.number_input(label, float(lo), float(hi), step=0.01 if "cgpa" in f else 0.5, key=f"in_{f}")
        st.form_submit_button("Predict", type="primary")
    row_in = {f: st.session_state[f"in_{f}"] for f in core.INPUT_FIELDS}
    row = core.clean_input(row_in)
    if row_in["team_min_cgpa"] > row_in["team_avg_cgpa"]:
        st.warning("Weakest member's CGPA cannot exceed the team average, so it was capped at the average.")
    elif row_in["group_size"] == 1 and row_in["team_min_cgpa"] != row_in["team_avg_cgpa"]:
        st.warning("Solo group: min CGPA was set equal to the average.")
    prob = core.predict_one(MODEL, row)
    tier = core.tier_of(prob, T_LOW, T_HIGH)
    with st.container(border=True):
        st.markdown(badge(tier, prob), unsafe_allow_html=True)
        st.markdown(f"**Predicted outcome:** {'POOR OUTCOME likely (fail / carry over)' if prob >= THRESHOLD else 'Likely to PASS'}"
                    f" &nbsp; (decision threshold {THRESHOLD:.2f})")
    st.subheader("Probability cut-offs and observed poor-outcome rate (test split)")
    tt = (test.groupby("tier")["poor_outcome_flag"].agg(groups="size", poor="sum", rate="mean").reindex(TIERS).reset_index())
    tt.insert(1, "range", [f"< {T_LOW:.2f}", f"{T_LOW:.2f} - {T_HIGH:.2f}", f">= {T_HIGH:.2f}"])
    a, b = st.columns(2)
    a.dataframe(tt.style.format({"rate": "{:.1%}"}), hide_index=True, width="stretch")
    a.caption("High = model threshold (F1-optimal on CV). Low = highest cut-off with <= 5% poor outcomes on CV.")
    fig = px.bar(tt, x="tier", y="rate", color="tier", color_discrete_map=TIER_COLOR, text=tt["rate"].map("{:.1%}".format))
    fig.update_layout(showlegend=False, yaxis_tickformat=".0%", height=340, title="Observed rate rises Low → Medium → High")
    b.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ A5
with tab5:
    st.header("A5 · Explainability (group in the A4 form)")
    prob, ex = core.explain(MODEL, row, REF)
    text, action = core.narrative(prob, tier, ex)
    st.markdown(badge(tier, prob), unsafe_allow_html=True)
    top = pd.concat([ex[ex["impact"] > 0.005].head(3), ex[ex["impact"] < -0.005].sort_values("impact").head(3)])
    l5, r5 = st.columns([3, 2])
    if len(top):
        top = top.assign(pts=top["impact"] * 100, name=[f"{r.label} = {core.fmt_val(r.feature, r.value)}" for r in top.itertuples()],
                         effect=["Raises risk" if v > 0 else "Lowers risk" for v in top["impact"]])
        fig = px.bar(top.iloc[::-1], x="pts", y="name", orientation="h", color="effect",
                     color_discrete_map={"Raises risk": "#c62828", "Lowers risk": "#2e7d32"},
                     labels={"pts": "Change in risk vs a typical group (pts)", "name": ""}, title="Top factors")
        l5.plotly_chart(fig, width="stretch")
    r5.markdown(text)
    r5.info(f"**Suggested support step:** {action}")

    st.divider()
    st.header("A5 · Fairness check (test split)")

    def ftable(col, min_n=0):
        rows = []
        for g, d in test.groupby(col):
            if len(d) < min_n:
                continue
            pos = d[d["poor_outcome_flag"] == 1]
            k = int(pos["predicted_poor"].sum())
            lo, hi = core.wilson(k, len(pos))
            rows.append({col: g, "Test groups": len(d), "Actual poor": len(pos), "Actual poor rate": d["poor_outcome_flag"].mean(),
                         "Predicted poor rate": d["predicted_poor"].mean(), "Recall": k / len(pos),
                         "Recall 95% CI": f"{lo:.0%} - {hi:.0%}", "_lo": lo, "_hi": hi})
        return pd.DataFrame(rows).set_index(col)

    fm = {"Actual poor rate": "{:.1%}", "Predicted poor rate": "{:.1%}", "Recall": "{:.1%}"}
    gender, campus = ftable("leader_gender"), ftable("campus", 30)
    st.dataframe(gender.drop(columns=["_lo", "_hi"]).style.format(fm), width="stretch")
    st.markdown(f"**Campuses with >= 30 test groups:** {len(campus)} of {test['campus'].nunique()}")
    st.dataframe(campus.sort_values("Recall").drop(columns=["_lo", "_hi"]).style.format(fm), width="stretch")
    gap = lambda t, c: (t[c].max() - t[c].min(), t[c].idxmax(), t[c].idxmin())
    g_rec, c_pred, c_rec = gap(gender, "Recall"), gap(campus, "Predicted poor rate"), gap(campus, "Recall")
    st.markdown(f"Largest campus gaps: predicted rate **{c_pred[0]:.1%}** ({c_pred[1]} vs {c_pred[2]}), "
                f"recall **{c_rec[0]:.1%}** ({c_rec[1]} vs {c_rec[2]}).")
    ov_g = gender["_lo"].max() <= gender["_hi"].min()
    ov_c = campus["_lo"].max() <= campus["_hi"].min()
    pr, ar = gender["Predicted poor rate"], gender["Actual poor rate"]
    st.success(f"**Takeaway:** Gender: {'no clear disparity; the' if ov_g else 'the'} recall gap of {g_rec[0]:.0%} "
               f"{'sits inside overlapping confidence intervals' if ov_g else 'should be investigated'} (only ~"
               f"{int(gender['Actual poor'].mean())} poor outcomes per gender). Female-led flagged {pr['Female']:.0%} (actual {ar['Female']:.0%}), "
               f"Male-led {pr['Male']:.0%} (actual {ar['Male']:.0%}). Campus: the {c_rec[0]:.0%} recall gap rests on only "
               f"{int(campus.loc[c_rec[1], 'Actual poor'])} and {int(campus.loc[c_rec[2], 'Actual poor'])} poor outcomes, so "
               f"{'the intervals overlap and it is not a confirmed disparity' if ov_c else 'it is worth monitoring'}. Confidence is low.")

# ------------------------------------------------------------------ A6
with tab6:
    st.header("A6 · Weekly Watchlist")
    K = st.slider("Coordinator capacity K", 10, 100, 30)
    top6 = core.watchlist(test, MODEL, REF, K)
    caught, base_rate = int(top6["poor_outcome_flag"].sum()), test["poor_outcome_flag"].mean()
    w = st.columns(3)
    w[0].metric("Truly poor groups captured", caught)
    w[1].metric("Precision at K", f"{caught / K:.0%}")
    w[2].metric("Expected by random picking", f"{K * base_rate:.1f}")
    show6 = top6[["group_id", "probability", "tier", "top_risk_driver"]].reset_index(drop=True)
    st.dataframe(show6.style.format({"probability": "{:.1%}"}), width="stretch")
    st.download_button("Download watchlist CSV", show6.to_csv(index=False), f"watchlist_top{K}.csv", "text/csv")

# ------------------------------------------------------------------ A7
with tab7:
    st.header("A7 · What-if Simulator (group in the A4 form)")
    st.warning("The model shows **association, not a guaranteed effect** of the change.")
    new = dict(row)
    lc = st.columns(len(core.LEVERS))
    for box, (f, (label, lo, hi, step, _t, _d)) in zip(lc, core.LEVERS.items()):
        if lo is None:
            new[f] = box.selectbox(label, ["No", "Yes"], index=["No", "Yes"].index(row[f]), key=f"lv_{f}")
        else:
            new[f] = box.slider(label, lo, hi, int(row[f]), step, key=f"lv_{f}")
    p1 = core.predict_one(MODEL, new)
    t1 = core.tier_of(p1, T_LOW, T_HIGH)
    _, act1 = core.narrative(p1, t1, core.explain(MODEL, new, REF)[1])
    b1, b2 = st.columns(2)
    with b1.container(border=True):
        st.markdown("**Before**"); st.markdown(badge(tier, prob), unsafe_allow_html=True); st.caption(f"Action: {action}")
    with b2.container(border=True):
        st.markdown("**After**"); st.markdown(badge(t1, p1), unsafe_allow_html=True); st.caption(f"Action: {act1}")
    st.metric("Change", f"{p1:.1%}", f"{(p1 - prob) * 100:+.1f} pts", delta_color="inverse")
    if st.button("Which lever moves this group's score most?", type="primary"):
        st.dataframe(core.best_lever(MODEL, row).style.format({"New probability": "{:.1%}", "Change (pts)": "{:+.1f}"}),
                     hide_index=True, width="stretch")
