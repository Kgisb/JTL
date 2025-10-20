
# JTL Enhanced App — Final
import os
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st
from datetime import date as _date, timedelta as _timedelta

st.set_page_config(page_title="JTL Enhanced App", layout="wide")

@st.cache_data(show_spinner=False)
def load_data():
    # Try to load the CSV shipped by the user
    path_opts = ["Master_sheet-DB.csv", "./Master_sheet-DB.csv", "/mnt/data/Master_sheet-DB.csv"]
    for p in path_opts:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                return df
            except Exception:
                pass
    st.warning("Could not find or open 'Master_sheet-DB.csv'. Place it next to the app.", icon="⚠️")
    return pd.DataFrame()

def _to_dt(series):
    try:
        s = pd.to_datetime(series, errors="coerce", infer_datetime_format=True, dayfirst=True)
    except Exception:
        s = pd.to_datetime(series, errors="coerce")
    mask = s.isna()
    if mask.any():
        raw = series.astype(str).str.strip()
        m2 = raw.str.fullmatch(r"\d{8}", na=False)
        s2 = pd.to_datetime(raw.where(m2), format="%d%m%Y", errors="coerce")
        s = s.fillna(s2)
        mask = s.isna()
        if mask.any():
            s3 = pd.to_datetime(raw.where(mask).str.slice(0, 10), errors="coerce", dayfirst=True)
            s = s.fillna(s3)
    return s

def _resolve_col(df, preferred, candidates):
    if isinstance(preferred, str) and preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    low = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        k = c.lower().strip()
        if k in low:
            return low[k]
    return None

def _norm(s): return s.fillna("Unknown").astype(str).str.strip()

def _month_bounds(d0: _date):
    from calendar import monthrange
    start = _date(d0.year, d0.month, 1)
    end = _date(d0.year, d0.month, monthrange(d0.year, d0.month)[1])
    return start, end

def date_window_controls(prefix):
    today = _date.today()
    scope = st.radio("Date scope", ["Today","Yesterday","This Month","Last Month","Custom"], index=2,
                     horizontal=True, key=f"{prefix}_scope")
    if scope == "Today":
        start_d, end_d = today, today
    elif scope == "Yesterday":
        start_d = today - _timedelta(days=1); end_d = start_d
    elif scope == "This Month":
        start_d, end_d = _month_bounds(today)
    elif scope == "Last Month":
        first_this = _date(today.year, today.month, 1)
        last_prev  = first_this - _timedelta(days=1)
        start_d, end_d = _month_bounds(last_prev)
    else:
        c1, c2 = st.columns(2)
        with c1: start_d = st.date_input("Start", value=_month_bounds(today)[0], key=f"{prefix}_start")
        with c2: end_d   = st.date_input("End", value=today, key=f"{prefix}_end")
        if end_d < start_d:
            start_d, end_d = end_d, start_d
    return start_d, end_d

df = load_data()

st.sidebar.title("Funnel & Movement")
pill = st.sidebar.radio(
    "Choose a pill",
    ["Closed Lost Analysis", "Booking Analysis", "Trial Trend"],
    index=0
)

# Common column guesses
counsellor_col = _resolve_col(df, None, ["Student/Academic Counsellor","Academic Counsellor","Counsellor","Counselor","Deal Owner"])
country_col    = _resolve_col(df, None, ["Country","Country Name"])
source_col     = _resolve_col(df, None, ["JetLearn Deal Source","Deal Source","Source","Original source"])
create_col     = _resolve_col(df, None, ["Create Date","Created Date","Deal Create Date","CreateDate","Created On","Creation Date"])

# =========================== Closed Lost Analysis ===========================
def render_closed_lost():
    st.header("Closed Lost Analysis")
    cl_col  = _resolve_col(df, None, ["[Deal Stage] - Closed Lost Trigger Date","Closed Lost Trigger Date","Closed Lost Date","Closed-Lost Trigger Date"])
    trial_s = _resolve_col(df, None, ["Trial Scheduled Date","Trial Schedule Date","Trial Booking Date","First Calibration Scheduled Date"])
    trial_r = _resolve_col(df, None, ["Trial Rescheduled Date","Trial Re-scheduled Date","Calibration Rescheduled Date"])
    cal_d   = _resolve_col(df, None, ["Calibration Done Date","Calibration Completed Date","Trial Done Date"])
    if cl_col is None or create_col is None:
        st.warning("Missing Closed Lost or Create Date columns.", icon="⚠️"); return

    d = df.copy()
    d["_CL"] = _to_dt(d[cl_col])
    d["_C"]  = _to_dt(d[create_col])
    d["_TS"] = _to_dt(d[trial_s]) if trial_s else pd.NaT
    d["_TR"] = _to_dt(d[trial_r]) if trial_r else pd.NaT
    d["_CD"] = _to_dt(d[cal_d]) if cal_d else pd.NaT
    d["_AC"] = _norm(d[counsellor_col]) if counsellor_col else pd.Series(["Unknown"]*len(d))
    d["_CNT"]=_norm(d[country_col])    if country_col    else pd.Series(["Unknown"]*len(d))
    d["_SRC"]=_norm(d[source_col])     if source_col     else pd.Series(["Unknown"]*len(d))

    c0,c1,c2,c3 = st.columns([1,1,1,2])
    with c0: mode = st.radio("Counting mode", ["MTD","Cohort"], index=0, horizontal=True, key="cl_mode")
    with c1: start_d, end_d = date_window_controls("cl")
    with c2: chart = st.radio("Chart", ["Stacked Bar","Line"], index=0, horizontal=True, key="cl_chart")
    with c3:
        dim_opts = []
        if counsellor_col: dim_opts.append("Academic Counsellor")
        if country_col:    dim_opts.append("Country")
        if source_col:     dim_opts.append("JetLearn Deal Source")
        sel_dims = st.multiselect("Dimensions (1–2 best)", options=dim_opts, default=(dim_opts[:1] if dim_opts else []))

    cl_in = d["_CL"].dt.date.between(start_d, end_d)
    in_scope = cl_in & d["_C"].dt.date.between(start_d, end_d) if mode=="MTD" else cl_in
    dd = d.loc[in_scope].copy()
    st.caption(f"Window: **{start_d} → {end_d}** • Mode: **{mode}** • Rows: **{len(dd)}**")

    if dd.empty:
        st.info("No Closed Lost rows in window."); return

    def _map_dim(x):
        return {"Academic Counsellor":"_AC","Country":"_CNT","JetLearn Deal Source":"_SRC"}.get(x)
    dim_cols = [_map_dim(x) for x in sel_dims if _map_dim(x)]
    if not dim_cols: dd["_All"]="All"; dim_cols=["_All"]

    agg = dd.groupby(dim_cols, dropna=False).size().reset_index(name="Closed Lost Count")
    if chart == "Stacked Bar":
        if len(dim_cols)==1:
            ch = alt.Chart(agg).mark_bar().encode(
                x=alt.X(f"{dim_cols[0]}:N", title=sel_dims[0] if sel_dims else "All"),
                y=alt.Y("Closed Lost Count:Q"),
                tooltip=[alt.Tooltip(f"{dim_cols[0]}:N"), alt.Tooltip("Closed Lost Count:Q")]
            ).properties(height=340)
        else:
            ch = alt.Chart(agg).mark_bar().encode(
                x=alt.X(f"{dim_cols[0]}:N", title=sel_dims[0]),
                y=alt.Y("Closed Lost Count:Q"),
                color=alt.Color(f"{dim_cols[1]}:N", title=sel_dims[1]),
                tooltip=[alt.Tooltip(f"{dim_cols[0]}:N"), alt.Tooltip(f"{dim_cols[1]}:N"), alt.Tooltip("Closed Lost Count:Q")]
            ).properties(height=340)
    else:
        dd["_d"] = dd["_CL"].dt.date
        ts = dd.groupby(["_d"] + dim_cols, dropna=False).size().reset_index(name="Closed Lost Count")
        color_enc = alt.Color(f"{dim_cols[0]}:N", title=(sel_dims[0] if sel_dims else "All")) if dim_cols else alt.value("steelblue")
        ch = alt.Chart(ts).mark_line(point=True).encode(
            x=alt.X("_d:T", title=None),
            y=alt.Y("Closed Lost Count:Q"),
            color=color_enc,
            tooltip=[alt.Tooltip("_d:T", title="Date"), alt.Tooltip("Closed Lost Count:Q")]
        ).properties(height=340)

    st.altair_chart(ch, use_container_width=True)
    st.dataframe(agg, use_container_width=True, hide_index=True)
    st.download_button("Download CSV — Closed Lost by Dimension", data=agg.to_csv(index=False).encode("utf-8"),
                       file_name="closed_lost_by_dimension.csv", mime="text/csv")

# =========================== Booking Analysis ===========================
def render_booking():
    st.header("Booking Analysis")
    trigger = _resolve_col(df, None, ["[Trigger] - Calibration Booking Date","Trigger - Calibration Booking Date","Calibration Booking Date"])
    slot    = _resolve_col(df, None, ["Calibration Slot (Deal)","Calibration Slot","Booking Slot (Deal)"])
    first   = _resolve_col(df, None, ["First Calibration Scheduled Date","Trial Scheduled Date","Trial Schedule Date"])
    if trigger is None: st.warning("Missing Booking Trigger date.", icon="⚠️"); return

    d = df.copy()
    d["_TRIG"] = _to_dt(d[trigger])
    d["_SLOT"] = d[slot].astype(str).str.strip() if slot else pd.Series([""]*len(d))
    d["_FIRST"]= _to_dt(d[first]) if first else pd.NaT
    d["_C"]    = _to_dt(d[create_col]) if create_col else pd.NaT

    pre_mask  = d["_SLOT"].notna() & (d["_SLOT"].str.len()>0) & (d["_SLOT"].str.lower()!="nan")
    self_mask = (~pre_mask) & d["_FIRST"].notna()
    d["_BKTYPE"] = np.select([pre_mask, self_mask], ["Pre-book","Self book"], default="Unknown")

    d["_AC"] = _norm(d[counsellor_col]) if counsellor_col else pd.Series(["Unknown"]*len(d))
    d["_CNT"]= _norm(d[country_col])    if country_col    else pd.Series(["Unknown"]*len(d))
    d["_SRC"]= _norm(d[source_col])     if source_col     else pd.Series(["Unknown"]*len(d))

    # Extra slice flags
    resch   = _resolve_col(df, None, ["Trial Rescheduled Date","Trial Re-scheduled Date","Calibration Rescheduled Date"])
    caldone = _resolve_col(df, None, ["Calibration Done Date","Calibration Completed Date","Trial Done Date"])
    enrol   = _resolve_col(df, None, ["Payment Received Date","Enrollment Date","Enrolment Date","Payment Date"])
    d["_HAS_FIRST"]   = pd.Series(pd.notna(d["_FIRST"]).map({True:"Yes", False:"No"}))
    d["_HAS_RESCH"]   = pd.Series(pd.notna(_to_dt(d[resch])).map({True:"Yes", False:"No"})) if resch else pd.Series(["No"]*len(d))
    d["_HAS_CALDONE"] = pd.Series(pd.notna(_to_dt(d[caldone])).map({True:"Yes", False:"No"})) if caldone else pd.Series(["No"]*len(d))
    d["_HAS_ENRL"]    = pd.Series(pd.notna(_to_dt(d[enrol])).map({True:"Yes", False:"No"})) if enrol else pd.Series(["No"]*len(d))

    c0,c1,c2,c3 = st.columns([1.0,1.0,1.0,1.6])
    with c0: mode = st.radio("Counting mode", ["MTD","Cohort"], index=0, horizontal=True, key="bk_mode")
    with c1: start_d, end_d = date_window_controls("bk")
    with c2: gran = st.radio("Granularity", ["Daily","Monthly"], index=0, horizontal=True, key="bk_gran")
    with c3:
        dims = st.multiselect("Slice by", options=[
            "Academic Counsellor","Country","JetLearn Deal Source","Booking Type",
            "First Trial","Trial Reschedule","Calibration Done","Enrolment"
        ], default=["Booking Type"], key="bk_dims")

    cl_in = d["_TRIG"].dt.date.between(start_d, end_d)
    in_win = cl_in & d["_C"].dt.date.between(start_d, end_d) if (mode=="MTD" and create_col) else cl_in
    dfw = d.loc[in_win].copy()

    # Filters
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        ac_opts = ["All"] + sorted(dfw["_AC"].unique().tolist())
        sel_ac = st.multiselect("Academic Counsellor", ac_opts, default=["All"])
    with c2:
        ctry_opts = ["All"] + sorted(dfw["_CNT"].unique().tolist())
        sel_cty = st.multiselect("Country", ctry_opts, default=["All"])
    with c3:
        src_opts = ["All"] + sorted(dfw["_SRC"].unique().tolist())
        sel_src = st.multiselect("JetLearn Deal Source", src_opts, default=["All"])
    with c4:
        bkt_opts = ["All","Pre-book","Self book","Unknown"]
        sel_bkt = st.multiselect("Booking Type", bkt_opts, default=["Pre-book","Self book"])

    def _resolve(vals, all_vals):
        return all_vals if ("All" in vals or not vals) else vals

    mask = (
        dfw["_AC"].isin(_resolve(sel_ac, sorted(dfw["_AC"].unique().tolist()))) &
        dfw["_CNT"].isin(_resolve(sel_cty, sorted(dfw["_CNT"].unique().tolist()))) &
        dfw["_SRC"].isin(_resolve(sel_src, sorted(dfw["_SRC"].unique().tolist()))) &
        dfw["_BKTYPE"].isin(_resolve(sel_bkt, ["Pre-book","Self book","Unknown"]))
    )
    dfw = dfw.loc[mask].copy()

    st.caption(f"Window: **{start_d} → {end_d}** • Mode: **{mode}** • Rows: **{len(dfw)}**")
    if dfw.empty:
        st.info("No records for selected filters/date range."); return

    dfw["_day"] = dfw["_TRIG"].dt.date
    dfw["_mon"] = pd.to_datetime(dfw["_TRIG"].dt.to_period("M").astype(str))

    def _map_dim(x):
        return {
            "Academic Counsellor":"_AC","Country":"_CNT","JetLearn Deal Source":"_SRC","Booking Type":"_BKTYPE",
            "First Trial":"_HAS_FIRST","Trial Reschedule":"_HAS_RESCH","Calibration Done":"_HAS_CALDONE","Enrolment":"_HAS_ENRL"
        }.get(x)
    dim_cols = [_map_dim(x) for x in dims if _map_dim(x)]
    if not dim_cols: dfw["_All"]="All"; dim_cols=["_All"]

    if gran=="Daily":
        grp = ["_day"] + dim_cols
        series = dfw.groupby(grp, dropna=False).size().rename("Count").reset_index()
        x_enc = alt.X("_day:T", title=None)
    else:
        grp = ["_mon"] + dim_cols
        series = dfw.groupby(grp, dropna=False).size().rename("Count").reset_index()
        x_enc = alt.X("_mon:T", title=None)

    chart = st.radio("Chart", ["Stacked Bar","Horizontal Bar","Line"], index=0, horizontal=True, key="bk_chart")
    if chart=="Stacked Bar":
        color_enc = alt.Color(f"{dim_cols[0]}:N", title=dims[0] if dims else "All")
        ch = alt.Chart(series).mark_bar().encode(x=x_enc, y=alt.Y("Count:Q"), color=color_enc, tooltip=[alt.Tooltip("Count:Q")]).properties(height=320)
    elif chart=="Horizontal Bar":
        y_enc = alt.Y("_day:T", title=None) if gran=="Daily" else alt.Y("_mon:T", title=None)
        color_enc = alt.Color(f"{dim_cols[0]}:N", title=dims[0] if dims else "All")
        ch = alt.Chart(series).mark_bar().encode(y=y_enc, x=alt.X("Count:Q"), color=color_enc, tooltip=[alt.Tooltip("Count:Q")]).properties(height=320)
    else:
        color_enc = alt.Color(f"{dim_cols[0]}:N", title=dims[0] if dims else "All")
        ch = alt.Chart(series).mark_line(point=True).encode(x=x_enc, y=alt.Y("Count:Q"), color=color_enc, tooltip=[alt.Tooltip("Count:Q")]).properties(height=320)
    st.altair_chart(ch, use_container_width=True)

    pretty = series.rename(columns={
        "_day":"Date","_mon":"Month","_AC":"Academic Counsellor","_CNT":"Country","_SRC":"JetLearn Deal Source","_BKTYPE":"Booking Type"
    })
    st.dataframe(pretty, use_container_width=True, hide_index=True)
    st.download_button("Download CSV — Booking Analysis Trend", data=pretty.to_csv(index=False).encode("utf-8"),
                       file_name="booking_analysis_trend.csv", mime="text/csv")

    # Comparison: Booking vs Trial Scheduled
    st.markdown("### Comparison: Booking vs Trial Scheduled")
    do_compare = st.checkbox("Show comparison vs Trial Scheduled", value=True, key="bk_compare_toggle")
    if do_compare:
        ts = d[d["_FIRST"].notna()].copy()
        ts_in = ts["_FIRST"].dt.date.between(start_d, end_d)
        if mode == "MTD" and create_col:
            ts_in = ts_in & ts["_C"].dt.date.between(start_d, end_d)
        ts = ts.loc[ts_in].copy()

        ts_mask = ts["_AC"].isin(_resolve(sel_ac, sorted(ts["_AC"].unique().tolist()))) & \
                  ts["_CNT"].isin(_resolve(sel_cty, sorted(ts["_CNT"].unique().tolist()))) & \
                  ts["_SRC"].isin(_resolve(sel_src, sorted(ts["_SRC"].unique().tolist())))
        ts = ts.loc[ts_mask].copy()
        if gran=="Daily":
            ts["_day"] = ts["_FIRST"].dt.date

        # Robust build regardless of granularity
        if gran=="Daily":
            ts["_day"] = ts["_FIRST"].dt.date
            ts_series = ts.groupby(["_day"], dropna=False).size().rename("Count").reset_index()
            ts_series["_x"] = ts_series["_day"]
            bk_series = series.copy()
            if "_day" in bk_series.columns: bk_series["_x"]=bk_series["_day"]
            else:
                tmp = dfw.groupby(["_day"], dropna=False).size().rename("Count").reset_index()
                bk_series = tmp.assign(_x=tmp["_day"])
        else:
            ts["_mon"] = pd.to_datetime(ts["_FIRST"].dt.to_period("M").astype(str))
            ts_series = ts.groupby(["_mon"], dropna=False).size().rename("Count").reset_index()
            ts_series["_x"] = ts_series["_mon"]
            bk_series = series.copy()
            if "_mon" in bk_series.columns: bk_series["_x"]=bk_series["_mon"]
            else:
                tmp = dfw.groupby(["_mon"], dropna=False).size().rename("Count").reset_index()
                bk_series = tmp.assign(_x=tmp["_mon"])

        ts_series["Series"] = "Trial Scheduled"
        bk_series_slim = bk_series[["_x","Count"]].copy(); bk_series_slim["Series"]="Booking Trigger"
        comp = pd.concat([bk_series_slim, ts_series[["_x","Count","Series"]]], ignore_index=True)
        comp = comp.sort_values(["_x","Series"]).reset_index(drop=True)
        ch_cmp = alt.Chart(comp).mark_line(point=True).encode(
            x=alt.X("_x:T", title=None), y=alt.Y("Count:Q"), color=alt.Color("Series:N", title="Series"),
            tooltip=[alt.Tooltip("_x:T", title="Date/Month"), alt.Tooltip("Series:N"), alt.Tooltip("Count:Q")]
        ).properties(height=320)
        st.altair_chart(ch_cmp, use_container_width=True)

        pretty_cmp = comp.rename(columns={"_x":"Date" if gran=="Daily" else "Month"})
        st.download_button("Download CSV — Booking vs Trial Scheduled", data=pretty_cmp.to_csv(index=False).encode("utf-8"),
                           file_name="booking_vs_trial_scheduled.csv", mime="text/csv")

# =========================== Trial Trend ===========================
def render_trial_trend():
    st.header("Trial Trend")
    first   = _resolve_col(df, None, ["First Calibration Scheduled Date","Trial Scheduled Date","Trial Schedule Date"])
    resch   = _resolve_col(df, None, ["Trial Rescheduled Date","Trial Re-scheduled Date","Calibration Rescheduled Date"])
    done    = _resolve_col(df, None, ["Calibration Done Date","Calibration Completed Date","Trial Done Date"])
    enrol   = _resolve_col(df, None, ["Payment Received Date","Enrollment Date","Enrolment Date","Payment Date"])

    d = df.copy()
    d["_FT"]   = _to_dt(d[first]) if first else pd.NaT
    d["_TR"]   = _to_dt(d[resch]) if resch else pd.NaT
    d["_TD"]   = _to_dt(d[done]) if done else pd.NaT
    d["_ENR"]  = _to_dt(d[enrol]) if enrol else pd.NaT
    d["_C"]    = _to_dt(d[create_col]) if create_col else pd.NaT
    d["_AC"] = _norm(d[counsellor_col]) if counsellor_col else pd.Series(["Unknown"]*len(d))
    d["_CNT"]= _norm(d[country_col])    if country_col    else pd.Series(["Unknown"]*len(d))
    d["_SRC"]= _norm(d[source_col])     if source_col     else pd.Series(["Unknown"]*len(d))

    c0,c1,c2 = st.columns([1.0,1.0,2.0])
    with c0: mode = st.radio("Counting mode", ["MTD","Cohort"], index=0, horizontal=True, key="tt_mode")
    with c1: start_d, end_d = date_window_controls("tt")
    with c2:
        dims = st.multiselect("Slice by", options=["Academic Counsellor","Country","JetLearn Deal Source"],
                              default=["Academic Counsellor"], key="tt_dims")
    gran = st.radio("Granularity", ["Daily","Monthly"], index=0, horizontal=True, key="tt_gran")

    events = []
    for _, row in d.iterrows():
        ac, cnt, src = row["_AC"], row["_CNT"], row["_SRC"]
        cdate = row["_C"]
        # Trial union per day
        trial_dates = set()
        if pd.notna(row["_FT"]): trial_dates.add(pd.Timestamp(row["_FT"]).normalize())
        if pd.notna(row["_TR"]): trial_dates.add(pd.Timestamp(row["_TR"]).normalize())
        for dt in sorted(trial_dates):
            events.append((dt, "Trial", ac, cnt, src, cdate))
        if pd.notna(row["_TD"]):
            events.append((pd.Timestamp(row["_TD"]).normalize(), "Trial Done", ac, cnt, src, cdate))
        if pd.notna(row["_ENR"]):
            events.append((pd.Timestamp(row["_ENR"]).normalize(), "Enrollment", ac, cnt, src, cdate))
        if pd.notna(cdate):
            events.append((pd.Timestamp(cdate).normalize(), "Lead", ac, cnt, src, cdate))

    if not events:
        st.info("No events present."); return

    ev = pd.DataFrame(events, columns=["_when","Metric","_AC","_CNT","_SRC","_C"])
    in_win = ev["_when"].dt.date.between(start_d, end_d)
    if mode=="MTD" and create_col:
        in_win = in_win & pd.to_datetime(ev["_C"]).dt.date.between(start_d, end_d)
    ev = ev.loc[in_win].copy()

    c1,c2,c3 = st.columns(3)
    with c1:
        ac_opts = ["All"] + sorted(ev["_AC"].unique().tolist())
        sel_ac = st.multiselect("Academic Counsellor", ac_opts, default=["All"], key="tt_ac")
    with c2:
        ctry_opts = ["All"] + sorted(ev["_CNT"].unique().tolist())
        sel_cty = st.multiselect("Country", ctry_opts, default=["All"], key="tt_cty")
    with c3:
        src_opts = ["All"] + sorted(ev["_SRC"].unique().tolist())
        sel_src = st.multiselect("JetLearn Deal Source", src_opts, default=["All"], key="tt_src")

    def _resolve(vals, all_vals):
        return all_vals if ("All" in vals or not vals) else vals

    ev = ev[ev["_AC"].isin(_resolve(sel_ac, sorted(ev["_AC"].unique().tolist()))) &
            ev["_CNT"].isin(_resolve(sel_cty, sorted(ev["_CNT"].unique().tolist()))) &
            ev["_SRC"].isin(_resolve(sel_src, sorted(ev["_SRC"].unique().tolist())))].copy()

    st.caption(f"Window: **{start_d} → {end_d}** • Mode: **{mode}** • Rows: **{len(ev)}**")
    if ev.empty:
        st.info("No events after filtering."); return

    ev["_day"] = ev["_when"].dt.date
    ev["_mon"] = pd.to_datetime(ev["_when"].dt.to_period("M").astype(str))

    def _map_dim(x): return {"Academic Counsellor":"_AC","Country":"_CNT","JetLearn Deal Source":"_SRC"}.get(x)
    dim_cols = [_map_dim(x) for x in dims if _map_dim(x)]
    if not dim_cols: ev["_All"]="All"; dim_cols=["_All"]

    if gran=="Daily":
        grp = ["_day"] + dim_cols + ["Metric"]
        ser = ev.groupby(grp, dropna=False).size().rename("Count").reset_index()
        x_enc = alt.X("_day:T", title=None)
    else:
        grp = ["_mon"] + dim_cols + ["Metric"]
        ser = ev.groupby(grp, dropna=False).size().rename("Count").reset_index()
        x_enc = alt.X("_mon:T", title=None)

    chart = st.radio("Chart", ["Stacked Bar","Horizontal Bar","Line"], index=0, horizontal=True, key="tt_chart")
    if chart=="Stacked Bar":
        ch = alt.Chart(ser).mark_bar().encode(x=x_enc, y=alt.Y("Count:Q"), color=alt.Color("Metric:N", title="Metric"),
                                              tooltip=[alt.Tooltip("Metric:N"), alt.Tooltip("Count:Q")]).properties(height=320)
    elif chart=="Horizontal Bar":
        y_enc = alt.Y("_day:T", title=None) if gran=="Daily" else alt.Y("_mon:T", title=None)
        ch = alt.Chart(ser).mark_bar().encode(y=y_enc, x=alt.X("Count:Q"), color=alt.Color("Metric:N", title="Metric"),
                                              tooltip=[alt.Tooltip("Metric:N"), alt.Tooltip("Count:Q")]).properties(height=320)
    else:
        ch = alt.Chart(ser).mark_line(point=True).encode(x=x_enc, y=alt.Y("Count:Q"), color=alt.Color("Metric:N", title="Metric"),
                                                        tooltip=[alt.Tooltip("Metric:N"), alt.Tooltip("Count:Q")]).properties(height=320)
    st.altair_chart(ch, use_container_width=True)

    # Pretty table
    pretty = ser.rename(columns={"_day":"Date","_mon":"Month","_AC":"Academic Counsellor","_CNT":"Country","_SRC":"JetLearn Deal Source"})
    st.dataframe(pretty, use_container_width=True, hide_index=True)
    st.download_button("Download CSV — Trial Trend", data=pretty.to_csv(index=False).encode("utf-8"),
                       file_name="trial_trend.csv", mime="text/csv")

    # % Trend
    st.markdown("### % Trend (B / A)")
    metric_opts = ["Lead","Trial","Trial Done","Enrollment"]
    den = st.selectbox("Metric A (denominator)", metric_opts, index=1, key="tt_den")
    num = st.selectbox("Metric B (numerator)", metric_opts, index=2, key="tt_num")

    time_col = "_day" if gran=="Daily" else "_mon"
    pivot = ser.pivot_table(index=time_col, columns="Metric", values="Count", aggfunc="sum", fill_value=0).reset_index()
    if den in pivot.columns and num in pivot.columns:
        pivot["Rate"] = np.where(pivot[den]>0, pivot[num]/pivot[den], np.nan)
        ch_rate = alt.Chart(pivot).mark_line(point=True).encode(
            x=alt.X(f"{time_col}:T", title=None),
            y=alt.Y("Rate:Q", axis=alt.Axis(format="%"), title=f"{num} / {den}"),
            tooltip=[alt.Tooltip(f"{time_col}:T", title="Time"), alt.Tooltip(den, title=den), alt.Tooltip(num, title=num), alt.Tooltip("Rate:Q", format=".2%")]
        ).properties(height=300)
        st.altair_chart(ch_rate, use_container_width=True)
        pretty_rate = pivot.rename(columns={time_col: ("Date" if gran=="Daily" else "Month")})
        st.dataframe(pretty_rate, use_container_width=True, hide_index=True)
        st.download_button("Download CSV — % Trend (B over A)", data=pretty_rate.to_csv(index=False).encode("utf-8"),
                           file_name="trial_percentage_trend.csv", mime="text/csv")
    else:
        st.info("Selected metrics have no data in the window.")

# Router
if pill == "Closed Lost Analysis":
    render_closed_lost()
elif pill == "Booking Analysis":
    render_booking()
else:
    render_trial_trend()
