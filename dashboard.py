"""
Generate an interactive HTML dashboard for PDB statistics from the annual per-method CSV.
"""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit


def sigmoid(x: np.ndarray, L: float, k: float, x0: float, b: float) -> np.ndarray:
    """Logistic sigmoid: L / (1 + exp(-k*(x - x0))) + b"""
    return L / (1 + np.exp(-k * (x - x0))) + b


def exponential(x: np.ndarray, a: float, k: float, x0: float, b: float) -> np.ndarray:
    """Exponential growth: a * exp(k*(x - x0)) + b"""
    return a * np.exp(k * (x - x0)) + b


def sigmoid_jacobian(x: np.ndarray, L: float, k: float, x0: float, b: float) -> np.ndarray:
    """Jacobian of sigmoid w.r.t. (L, k, x0, b) at each x. Returns shape (n, 4)."""
    exp_term = np.exp(-k * (x - x0))
    denom = 1 + exp_term
    dL = 1 / denom
    dk = L * (x - x0) * exp_term / (denom**2)
    dx0 = -L * k * exp_term / (denom**2)
    db = np.ones_like(x)
    return np.column_stack([dL, dk, dx0, db])


def confidence_band(
    x: np.ndarray, popt: np.ndarray, pcov: np.ndarray, z: float = 1.96
) -> tuple[np.ndarray, np.ndarray]:
    """95% confidence interval for the fit: y ± z*SE. z=1.96 for 95% CI."""
    jac = sigmoid_jacobian(x, *popt)
    se = np.sqrt(np.einsum("ni,nj,ij->n", jac, jac, pcov))
    y_fit = sigmoid(x, *popt)
    return y_fit - z * se, y_fit + z * se


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB to rgba(r,g,b,alpha)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def build_dashboard(
    csv_path: str | Path | None = None,
    output_path: str | Path | None = None,
    milestones_path: str | Path | None = None,
    temperatures_path: str | Path | None = None,
) -> None:
    """
    Load the PDB annual statistics CSV and build an interactive HTML dashboard
    with charts and a data table.

    Args:
        csv_path: Path to the CSV file. Defaults to the Sheet1 file in the same directory.
        output_path: Path for the output HTML file. Defaults to pdb_dashboard.html.
        milestones_path: Path to milestones CSV (Year, Milestone). Defaults to milestones.csv.
        temperatures_path: Path to temperature depositions CSV. Defaults to depositions_by_temperature.csv.
    """
    if csv_path is None:
        csv_path = Path(__file__).parent / "PDB Annual per method - Sheet1.csv"
    if output_path is None:
        output_path = Path(__file__).parent / "pdb_dashboard.html"
    if milestones_path is None:
        milestones_path = Path(__file__).parent / "milestones.csv"
    if temperatures_path is None:
        temperatures_path = Path(__file__).parent / "depositions_by_temperature.csv"

    df = pd.read_csv(csv_path)
    milestones_df = pd.DataFrame(columns=["Year", "Milestone"])
    if Path(milestones_path).exists():
        try:
            _m = pd.read_csv(milestones_path)
            if not _m.empty and "Year" in _m.columns and "Milestone" in _m.columns:
                milestones_df = _m
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            pass

    # Remove commas from numeric columns and convert to int
    numeric_cols = [
        "Total Number of Entries Available X-ray",
        "Number of Structures Released Annually X-ray",
        "Total Number of Entries Available cryoEM",
        "Number of Structures Released Annually cryoEM",
    ]
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(",", "").astype(int)

    # Shorter column names for display
    df_display = df.copy()
    df_display.columns = [
        "Year",
        "Total X-ray",
        "Annual X-ray",
        "Total cryoEM",
        "Annual cryoEM",
    ]

    # Shared colors for consistent styling
    xray_color = "#ff7f0e"
    cryoem_color = "#1f77b4"
    room_temp_xray_color = "#ff4500"  # rgb(255,69,0); same hue as Room temp X-ray stack fill
    room_temp_xray_fill = _hex_to_rgba(room_temp_xray_color, 0.6)
    today = date.today()
    current_year = today.year
    days_in_year = date(current_year, 12, 31).timetuple().tm_yday
    elapsed_fraction = today.timetuple().tm_yday / days_in_year

    # Annual chart: precompute fits, then add filled surfaces first (background) and lines last (foreground).
    years_arr = np.arange(df["Year"].min(), 2041, dtype=float)
    years_list = years_arr.tolist()
    years_extended = years_list  # embedded JS (sigmoid / exp x-domain)

    sigmoid_params: dict = {}
    sigmoid_trace_indices: dict[str, int] = {}
    sigmoid_prep: dict[str, dict | None] = {}

    for key, col, color, legendgroup in [
        ("xray", "Number of Structures Released Annually X-ray", xray_color, "X-ray-sigmoid"),
        ("cryoem", "Number of Structures Released Annually cryoEM", cryoem_color, "cryoEM-sigmoid"),
    ]:
        df_fit = df[df["Year"] < current_year]
        x_data = df_fit["Year"].values.astype(float)
        y_data = df_fit[col].values.astype(float)
        sigmoid_prep[key] = None
        try:
            p0 = (y_data.max() - y_data.min(), 0.1, x_data.mean(), y_data.min())
            popt, pcov = curve_fit(sigmoid, x_data, y_data, p0=p0, maxfev=20000)
            L, k, x0, b = popt
            y_fit = sigmoid(years_arr, *popt)
            y_fit = np.maximum(y_fit, 0)
            y_lo, y_hi = confidence_band(years_arr, popt, pcov)
            y_lo = np.maximum(y_lo, 0)
            y_hi = np.maximum(y_hi, 0)
            model_group = "xray-sigmoid-model" if key == "xray" else "cryoem-sigmoid-model"
            legend_name = "X-ray sigmoid fit" if key == "xray" else "cryoEM sigmoid fit"
            sigmoid_prep[key] = {
                "popt": popt,
                "pcov": pcov,
                "y_fit": y_fit,
                "y_lo": y_lo,
                "y_hi": y_hi,
                "color": color,
                "legendgroup": legendgroup,
                "model_group": model_group,
                "legend_name": legend_name,
            }
            sigmoid_params[key] = {
                "L": float(L),
                "k": float(k),
                "x0": float(x0),
                "b": float(b),
                "L_min": max(0, L * 0.1),
                "L_max": L * 2,
                "k_min": 0.001,
                "k_max": 1,
                "x0_min": 1975,
                "x0_max": 2040,
                "b_min": max(-5000, b - 3000),
                "b_max": min(10000, b + 3000),
            }
        except (RuntimeError, ValueError):
            pass

    exp_prep: dict | None = None
    sp_c = sigmoid_prep.get("cryoem")
    if sp_c is not None:
        df_fit = df[df["Year"] < current_year]
        cryo_col_name = "Number of Structures Released Annually cryoEM"
        df_fit_c = df_fit[df_fit[cryo_col_name] > 0]
        if len(df_fit_c) >= 4:
            x_data_c = df_fit_c["Year"].values.astype(float)
            y_data_c = df_fit_c[cryo_col_name].values.astype(float)
            try:
                p0_exp = (y_data_c.max(), 0.2, x_data_c.min(), 0.0)
                popt_exp, _ = curve_fit(
                    exponential,
                    x_data_c,
                    y_data_c,
                    p0=p0_exp,
                    maxfev=20000,
                    bounds=([0, 0, 1970, -1000], [1e6, 2, 2050, 10000]),
                )
                a, k_e, x0_exp, b_exp = popt_exp
                y_exp = np.maximum(exponential(years_arr, *popt_exp), 0)
                exp_prep = {
                    "popt": popt_exp,
                    "y_exp": y_exp,
                    "a": float(a),
                    "k": float(k_e),
                    "x0": float(x0_exp),
                    "b": float(b_exp),
                }
                sigmoid_params["cryoem_exp"] = {
                    "a": float(a),
                    "k": float(k_e),
                    "x0": float(x0_exp),
                    "b": float(b_exp),
                    "a_min": max(0.1, a * 0.1),
                    "a_max": a * 3,
                    "k_min": 0.01,
                    "k_max": 1,
                    "x0_min": 1970,
                    "x0_max": 2050,
                    "b_min": max(-1000, b_exp - 500),
                    "b_max": min(5000, b_exp + 500),
                }
            except (RuntimeError, ValueError):
                exp_prep = None

    df_merged = None
    room_col = cryo_col = None
    if Path(temperatures_path).exists():
        try:
            df_temp = pd.read_csv(temperatures_path)
            if not df_temp.empty and "Year" in df_temp.columns:
                room_col = [c for c in df_temp.columns if "273" in c or "Room" in c.lower()]
                cryo_col = [c for c in df_temp.columns if "150" in c or "Cryo" in c.lower()]
                if room_col and cryo_col:
                    df_merged = df[["Year", "Number of Structures Released Annually X-ray"]].merge(
                        df_temp[["Year", room_col[0], cryo_col[0]]],
                        on="Year",
                        how="inner",
                    )
        except (pd.errors.EmptyDataError, pd.errors.ParserError, KeyError):
            pass

    annual_traces: list = []
    trace_idx = 0
    ytd_lbl = f"to date ({current_year})"
    ye_lbl = f"year-end estimate ({current_year})"
    room_proj_lbl = f"Room temp X-ray ({current_year} projected)"
    df_curr_year = df[df["Year"] == current_year]
    has_curr_year = not df_curr_year.empty
    row_rt_curr = (
        df_merged[df_merged["Year"] == current_year]
        if df_merged is not None and room_col
        else pd.DataFrame()
    )
    has_room_proj = (
        has_curr_year
        and not row_rt_curr.empty
        and room_col
        and elapsed_fraction > 0
        and np.isfinite(float(row_rt_curr.iloc[0][room_col[0]]))
        and float(row_rt_curr.iloc[0][room_col[0]]) >= 0
    )

    def _annual_append(tr: go.Scatter) -> None:
        nonlocal trace_idx
        annual_traces.append(tr)
        trace_idx += 1

    # Legend order: lower legendrank = higher in legend. Use spaced ranks so each
    # legendgroup's minimum sorts in the requested order (Plotly groups by legendgroup,
    # orders groups by min(legendrank), then sorts within a group by legendrank).
    # Requires layout.legend.traceorder "grouped" — "normal" follows data order and ignores ranks.
    LR_X_MAIN, LR_X_YTD, LR_X_YE = 100, 110, 120
    LR_X_SIG = 130
    LR_RT_ROOM, LR_RT_PROJ, LR_RT_CRYO, LR_RT_OTH = 140, 150, 160, 170
    LR_E_MAIN, LR_E_YTD, LR_E_YE = 200, 210, 220
    LR_E_SIG, LR_E_EXP = 230, 240

    # --- Background: sigmoid CI fills (drawn first) ---
    for key in ("xray", "cryoem"):
        sp = sigmoid_prep.get(key)
        if sp is None:
            continue
        color = sp["color"]
        fill_color = _hex_to_rgba(color, 0.25)
        mg = sp["model_group"]
        _annual_append(
            go.Scatter(
                x=years_list,
                y=sp["y_lo"].tolist(),
                name=sp["legend_name"],
                legendgroup=mg,
                showlegend=False,
                mode="lines",
                line=dict(width=0),
                legendrank=5000,
                hoverinfo="skip",
            ),
        )
        _annual_append(
            go.Scatter(
                x=years_list,
                y=sp["y_hi"].tolist(),
                name=sp["legend_name"],
                legendgroup=mg,
                showlegend=False,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=fill_color,
                legendrank=5000,
                hoverinfo="skip",
            ),
        )

    # --- Background: temperature stacks ---
    n_temp_traces = 0
    if df_merged is not None and room_col and cryo_col:
        xray_annual = df_merged["Number of Structures Released Annually X-ray"].values
        room_temp = df_merged[room_col[0]].values
        cryogenic = df_merged[cryo_col[0]].values
        others = np.maximum(0, xray_annual - room_temp - cryogenic)
        stackgroup = "xray_temp"
        temp_specs = [
            (room_temp, "Room temp X-ray", room_temp_xray_fill, LR_RT_ROOM),
            (cryogenic, "Cryogenic X-ray", "rgba(0,255,255,0.6)", LR_RT_CRYO),
            (others, "Others", "rgba(136,136,136,0.6)", LR_RT_OTH),
        ]
        for y_vals, tname, fillcolor, rank_lr in temp_specs:
            _annual_append(
                go.Scatter(
                    x=df_merged["Year"],
                    y=y_vals,
                    name=tname,
                    legendgroup="xray-temp",
                    stackgroup=stackgroup,
                    mode="lines",
                    line=dict(width=0),
                    fillcolor=fillcolor,
                    showlegend=True,
                    legendrank=rank_lr,
                ),
            )
        n_temp_traces = 3

    def add_annual_method_traces(
        col: str,
        name: str,
        color: str,
        legendgroup: str,
        lr_main: int,
        lr_ytd: int,
        lr_ye: int,
    ) -> None:
        df_history = df[df["Year"] < current_year].sort_values("Year")
        _annual_append(
            go.Scatter(
                x=df_history["Year"],
                y=df_history[col],
                name=name,
                legendgroup=legendgroup,
                showlegend=True,
                legendrank=lr_main,
                mode="lines+markers",
                line=dict(width=2, color=color),
                marker=dict(color=color),
            ),
        )

        if not has_curr_year:
            return

        current_value = int(df_curr_year.iloc[0][col])
        projected_value = current_value / elapsed_fraction
        _annual_append(
            go.Scatter(
                x=[current_year],
                y=[current_value],
                name=ytd_lbl,
                legendgroup=legendgroup,
                showlegend=True,
                legendrank=lr_ytd,
                mode="markers",
                marker=dict(color=color, size=12, symbol="triangle-up", line=dict(color=color, width=2)),
            ),
        )
        _annual_append(
            go.Scatter(
                x=[int(df_history.iloc[-1]["Year"]), current_year] if not df_history.empty else [current_year],
                y=[int(df_history.iloc[-1][col]), projected_value] if not df_history.empty else [projected_value],
                legendgroup=legendgroup,
                mode="lines",
                line=dict(width=2, color=color),
                showlegend=False,
            ),
        )
        _annual_append(
            go.Scatter(
                x=[current_year],
                y=[projected_value],
                name=ye_lbl,
                legendgroup=legendgroup,
                showlegend=True,
                legendrank=lr_ye,
                mode="markers",
                marker=dict(
                    color=color,
                    size=12,
                    symbol="circle-open",
                    line=dict(color=color, width=2),
                ),
            ),
        )

    add_annual_method_traces(
        "Number of Structures Released Annually X-ray",
        "X-ray",
        xray_color,
        "xray-annual",
        LR_X_MAIN,
        LR_X_YTD,
        LR_X_YE,
    )
    add_annual_method_traces(
        "Number of Structures Released Annually cryoEM",
        "cryoEM",
        cryoem_color,
        "cryoem-annual",
        LR_E_MAIN,
        LR_E_YTD,
        LR_E_YE,
    )

    # Room-temp X-ray: run-rate year-end estimate (matches main X-ray projection assumption)
    if has_room_proj:
        room_ytd = float(row_rt_curr.iloc[0][room_col[0]])
        room_ye = room_ytd / elapsed_fraction
        _annual_append(
            go.Scatter(
                x=[current_year],
                y=[room_ye],
                name=room_proj_lbl,
                legendgroup="xray-temp",
                showlegend=True,
                legendrank=LR_RT_PROJ,
                mode="markers",
                marker=dict(
                    symbol="circle-open",
                    size=12,
                    color=room_temp_xray_color,
                    line=dict(width=2, color=room_temp_xray_color),
                ),
                hovertemplate=(
                    "Room temp X-ray, year-end estimate (run rate)<br>"
                    "Year=%{x}<br>structures ≈ %{y:,.0f}<extra></extra>"
                ),
            ),
        )

    # --- Foreground: model lines (drawn on top of annual lines) ---
    for key, col, color, legendgroup in [
        ("xray", "Number of Structures Released Annually X-ray", xray_color, "X-ray-sigmoid"),
        ("cryoem", "Number of Structures Released Annually cryoEM", cryoem_color, "cryoEM-sigmoid"),
    ]:
        sp = sigmoid_prep.get(key)
        if sp is None:
            continue
        y_fit_list = np.maximum(sp["y_fit"], 0).tolist()
        _annual_append(
            go.Scatter(
                x=years_list,
                y=y_fit_list,
                name=sp["legend_name"],
                legendgroup=sp["model_group"],
                showlegend=True,
                legendrank=LR_X_SIG if key == "xray" else LR_E_SIG,
                mode="lines",
                line=dict(width=2, color=color, dash="dash"),
            ),
        )
        _annual_append(
            go.Scatter(
                x=years_list,
                y=y_fit_list,
                name="Adjusted",
                legendgroup=f"{legendgroup}-adj",
                mode="lines",
                line=dict(width=2, color=color),
                visible=False,
                showlegend=False,
            ),
        )
        sigmoid_trace_indices[key] = trace_idx - 1

        if key == "xray":
            y_fit_arr = np.maximum(sp["y_fit"], 0)
            L_mfx, k_mfx, x0_mfx, b_mfx = 2000.0, 0.71, 2030.0, 0.0
            y_mfx = sigmoid(years_arr, L_mfx, k_mfx, x0_mfx, b_mfx)
            y_mfx = np.maximum(y_mfx, 0)
            y_xray_plus_mfx = (y_fit_arr + y_mfx).tolist()
            _annual_append(
                go.Scatter(
                    x=years_list,
                    y=y_xray_plus_mfx,
                    name="X-ray + MFX",
                    legendgroup="xray-mfx-adj",
                    mode="lines",
                    line=dict(width=2, color=xray_color, dash="dot"),
                    visible=False,
                    showlegend=False,
                ),
            )
            sigmoid_params["mfx"] = {
                "L": L_mfx,
                "k": k_mfx,
                "x0": x0_mfx,
                "b": b_mfx,
                "L_min": 0,
                "L_max": 5000,
                "k_min": 0.001,
                "k_max": 1,
                "x0_min": 2010,
                "x0_max": 2040,
                "b_min": -500,
                "b_max": 500,
            }
            sigmoid_trace_indices["xray_plus_mfx"] = trace_idx - 1

        if key == "cryoem" and exp_prep is not None:
            y_exp_list = np.maximum(exp_prep["y_exp"], 0).tolist()
            _annual_append(
                go.Scatter(
                    x=years_list,
                    y=y_exp_list,
                    name="cryoEM exponential fit",
                    legendgroup="cryoem-exp-model",
                    showlegend=True,
                    legendrank=LR_E_EXP,
                    mode="lines",
                    line=dict(width=2, color=cryoem_color, dash="dot"),
                ),
            )
            _annual_append(
                go.Scatter(
                    x=years_list,
                    y=y_exp_list,
                    name="Exponential adjusted",
                    legendgroup="cryoEM-exp-adj",
                    mode="lines",
                    line=dict(width=2, color=cryoem_color),
                    visible=False,
                    showlegend=False,
                ),
            )
            sigmoid_trace_indices["cryoem_exp_fit"] = trace_idx - 2
            sigmoid_trace_indices["cryoem_exp_adj"] = trace_idx - 1

    fig_annual = go.Figure(data=annual_traces)

    manual_fit_trace_indices = [
        sigmoid_trace_indices[k]
        for k in ("xray", "cryoem", "xray_plus_mfx", "cryoem_exp_adj")
        if k in sigmoid_trace_indices
    ]

    # Milestone shapes and annotations
    shapes = []
    annotations = []
    for _, row in milestones_df.iterrows():
        try:
            year = int(row["Year"])
            label = str(row["Milestone"]).strip() or str(year)
        except (ValueError, TypeError):
            continue
        shapes.append(
            dict(
                type="line",
                x0=year,
                x1=year,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color="#888", width=1, dash="dot"),
            )
        )
        annotations.append(
            dict(
                x=year,
                y=0.98,
                xref="x",
                yref="paper",
                text=label,
                showarrow=False,
                yanchor="top",
                font=dict(size=10, color="#555"),
                xanchor="center",
                textangle=-35,
            )
        )

    # Annual chart layout (single legend: data + model fits)
    fig_annual.update_layout(
        title="PDB Structures Released Annually by Method",
        xaxis_title="Year",
        yaxis_title="Structures Released Annually",
        template="plotly_white",
        height=720,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            tracegroupgap=6,
            traceorder="grouped",
        ),
        margin=dict(t=80, r=220, b=60),
        shapes=shapes,
        annotations=annotations,
    )
    fig_annual.update_xaxes(range=[df["Year"].min() - 1, 2041])
    fig_annual.update_yaxes(range=[0, 15000])

    # Total entries chart
    fig_total = go.Figure()
    fig_total.add_trace(
        go.Scatter(
            x=df["Year"],
            y=df["Total Number of Entries Available X-ray"],
            name="X-ray",
            legendgroup="X-ray",
            mode="lines+markers",
            line=dict(width=2, color=xray_color),
            marker=dict(color=xray_color),
        ),
    )
    fig_total.add_trace(
        go.Scatter(
            x=df["Year"],
            y=df["Total Number of Entries Available cryoEM"],
            name="cryoEM",
            legendgroup="cryoEM",
            mode="lines+markers",
            line=dict(width=2, color=cryoem_color),
            marker=dict(color=cryoem_color),
        ),
    )
    fig_total.update_layout(
        title="Total PDB Entries Available by Method",
        xaxis_title="Year",
        yaxis_title="Total Entries",
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80),
        shapes=shapes,
        annotations=annotations,
    )

    annual_html = fig_annual.to_html(full_html=False, include_plotlyjs="cdn", div_id="annual-chart")
    total_html = fig_total.to_html(full_html=False, include_plotlyjs=False, div_id="total-chart")

    # Build table with all plot-relevant columns
    df_table = df_display.copy()
    if Path(temperatures_path).exists():
        try:
            df_temp = pd.read_csv(temperatures_path)
            if not df_temp.empty and "Year" in df_temp.columns:
                room_col = [c for c in df_temp.columns if "273" in c or "Room" in c.lower()]
                cryo_col = [c for c in df_temp.columns if "150" in c or "Cryo" in c.lower()]
                if room_col and cryo_col:
                    df_table = df_table.merge(
                        df_temp[["Year", room_col[0], cryo_col[0]]],
                        on="Year",
                        how="left",
                    )
                    df_table["Others"] = np.maximum(
                        0,
                        df_table["Annual X-ray"].fillna(0) - df_table[room_col[0]].fillna(0) - df_table[cryo_col[0]].fillna(0),
                    )
                    df_table = df_table.rename(columns={
                        room_col[0]: "Room temp X-ray",
                        cryo_col[0]: "Cryogenic X-ray",
                    })
                    # Reorder: Year, Annual X-ray, Room temp X-ray, Cryogenic X-ray, Others, Total X-ray, Annual cryoEM, Total cryoEM
                    cols = ["Year", "Annual X-ray", "Room temp X-ray", "Cryogenic X-ray", "Others", "Total X-ray", "Annual cryoEM", "Total cryoEM"]
                    df_table = df_table[[c for c in cols if c in df_table.columns]]
        except (pd.errors.EmptyDataError, pd.errors.ParserError, KeyError):
            pass

    table_fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(df_table.columns),
                    fill_color="paleturquoise",
                    align="left",
                ),
                cells=dict(
                    values=[df_table[col].astype(str) for col in df_table.columns],
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )
    table_fig.update_layout(
        title="Data Table",
        height=400,
        margin=dict(t=60),
    )
    table_html = table_fig.to_html(full_html=False, include_plotlyjs=False)

    # Parameter metadata: icon, label, tooltip
    param_info = {
        "L": ("↕", "Peak", "Maximum curve height"),
        "k": ("↗", "Steepness", "Growth rate at midpoint"),
        "x0": ("📍", "Midpoint", "Year of inflection point"),
        "b": ("▬", "Floor", "Baseline / minimum value"),
    }
    param_info_exp = {
        "a": ("↕", "Scale", "Exponential amplitude"),
        "k": ("↗", "Rate", "Growth rate (1/year)"),
        "x0": ("📍", "Ref year", "Reference year"),
        "b": ("▬", "Floor", "Baseline offset"),
    }

    # Build sigmoid controls HTML and JS
    controls_html = ""
    controls_script = ""
    if sigmoid_params:
        controls_html = (
            '<div class="manual-fit"><h3>Manual fit</h3>'
            '<p class="manual-fit-toolbar"><label class="manual-fit-toggle">'
            '<input type="checkbox" id="show-manual-fit-on-plot"> Show on plot</label></p>'
            '<p class="manual-fit-hint">Sliders update manual overlays (sigmoid, MFX, exponential). '
            "Turn on <strong>Show on plot</strong> to display adjusted curves on the annual chart. "
            "Use the chart <strong>legend</strong> to toggle data series, sigmoid fits (with 95% CI), and the exponential fit.</p>"
            '<div class="param-groups">'
        )
        for key, label in [("xray", "X-ray"), ("mfx", "MFX effect"), ("cryoem", "cryoEM"), ("cryoem_exp", "cryoEM exponential")]:
            if key not in sigmoid_params:
                continue
            p = sigmoid_params[key]
            method_color = xray_color if key in ("xray", "mfx") else cryoem_color
            params_to_use = ["a", "k", "x0", "b"] if key == "cryoem_exp" else ["L", "k", "x0", "b"]
            info_dict = param_info_exp if key == "cryoem_exp" else param_info
            controls_html += f'<div class="param-group"><h4><span class="method-label" style="color: {method_color}">●</span> {label}</h4><div class="sliders">'
            for param in params_to_use:
                icon, param_label, tooltip = info_dict[param]
                v, vmin, vmax = p[param], p[f"{param}_min"], p[f"{param}_max"]
                step = (vmax - vmin) / 200 if vmax != vmin else 0.01
                controls_html += f'''
                <label class="slider-row" title="{tooltip}">
                    <span class="param-icon">{icon}</span>
                    <span class="param-name">{param_label}</span>
                    <span class="param-val" id="{key}-{param}-val">{v:.2f}</span>
                    <input type="range" id="{key}-{param}" min="{vmin}" max="{vmax}" step="{step}" value="{v}" class="slider">
                </label>'''
            controls_html += "</div></div>"
        controls_html += "</div></div>"

        controls_script = f"""
    <script>
    const sigmoidParams = {json.dumps(sigmoid_params)};
    const sigmoidTraceIndices = {json.dumps(sigmoid_trace_indices)};
    const manualFitTraceIndices = {json.dumps(manual_fit_trace_indices)};
    const yearsExtended = {json.dumps(years_extended)};

    function sigmoid(x, L, k, x0, b) {{
        return L / (1 + Math.exp(-k * (x - x0))) + b;
    }}

    function exponential(x, a, k, x0, b) {{
        return a * Math.exp(k * (x - x0)) + b;
    }}

    let xrayAdjustedY = null;

    function updateXrayPlusMfx() {{
        const gd = document.getElementById('annual-chart');
        if (!gd || typeof Plotly === 'undefined' || !sigmoidTraceIndices.xray_plus_mfx) return;
        if (xrayAdjustedY === null) {{
            const traceY = gd.data[sigmoidTraceIndices.xray]?.y;
            xrayAdjustedY = Array.isArray(traceY) ? traceY.slice() : yearsExtended.map(() => 0);
        }}
        const m = sigmoidParams.mfx;
        if (!m) return;
        const L = parseFloat(document.getElementById('mfx-L').value);
        const k = parseFloat(document.getElementById('mfx-k').value);
        const x0 = parseFloat(document.getElementById('mfx-x0').value);
        const b = parseFloat(document.getElementById('mfx-b').value);
        const yMfx = yearsExtended.map(x => Math.max(0, sigmoid(x, L, k, x0, b)));
        const ySum = xrayAdjustedY.map((v, i) => v + yMfx[i]);
        Plotly.restyle(gd, {{ y: [ySum] }}, [sigmoidTraceIndices.xray_plus_mfx]);
    }}

    function updateSigmoid(key) {{
        const p = sigmoidParams[key];
        const L = parseFloat(document.getElementById(key + '-L').value);
        const k = parseFloat(document.getElementById(key + '-k').value);
        const x0 = parseFloat(document.getElementById(key + '-x0').value);
        const b = parseFloat(document.getElementById(key + '-b').value);
        p.L = L; p.k = k; p.x0 = x0; p.b = b;
        document.getElementById(key + '-L-val').textContent = L.toFixed(2);
        document.getElementById(key + '-k-val').textContent = k.toFixed(2);
        document.getElementById(key + '-x0-val').textContent = x0.toFixed(2);
        document.getElementById(key + '-b-val').textContent = b.toFixed(2);
        const y = yearsExtended.map(x => Math.max(0, sigmoid(x, L, k, x0, b)));
        const traceIdx = sigmoidTraceIndices[key];
        const gd = document.getElementById('annual-chart');
        if (gd && typeof Plotly !== 'undefined') {{
            Plotly.restyle(gd, {{ y: [y] }}, [traceIdx]);
            if (key === 'xray') {{
                xrayAdjustedY = y.slice();
                updateXrayPlusMfx();
            }}
        }}
    }}

    function updateMfx() {{
        const p = sigmoidParams.mfx;
        if (!p) return;
        const L = parseFloat(document.getElementById('mfx-L').value);
        const k = parseFloat(document.getElementById('mfx-k').value);
        const x0 = parseFloat(document.getElementById('mfx-x0').value);
        const b = parseFloat(document.getElementById('mfx-b').value);
        p.L = L; p.k = k; p.x0 = x0; p.b = b;
        document.getElementById('mfx-L-val').textContent = L.toFixed(2);
        document.getElementById('mfx-k-val').textContent = k.toFixed(2);
        document.getElementById('mfx-x0-val').textContent = x0.toFixed(2);
        document.getElementById('mfx-b-val').textContent = b.toFixed(2);
        updateXrayPlusMfx();
    }}

    function updateCryoemExp() {{
        const p = sigmoidParams.cryoem_exp;
        if (!p) return;
        const a = parseFloat(document.getElementById('cryoem_exp-a').value);
        const k = parseFloat(document.getElementById('cryoem_exp-k').value);
        const x0 = parseFloat(document.getElementById('cryoem_exp-x0').value);
        const b = parseFloat(document.getElementById('cryoem_exp-b').value);
        p.a = a; p.k = k; p.x0 = x0; p.b = b;
        document.getElementById('cryoem_exp-a-val').textContent = a.toFixed(2);
        document.getElementById('cryoem_exp-k-val').textContent = k.toFixed(2);
        document.getElementById('cryoem_exp-x0-val').textContent = x0.toFixed(2);
        document.getElementById('cryoem_exp-b-val').textContent = b.toFixed(2);
        const y = yearsExtended.map(x => Math.max(0, exponential(x, a, k, x0, b)));
        const gd = document.getElementById('annual-chart');
        const expIdx = sigmoidTraceIndices.cryoem_exp_adj ?? sigmoidTraceIndices.cryoem_exp;
        if (gd && typeof Plotly !== 'undefined' && expIdx !== undefined) {{
            Plotly.restyle(gd, {{ y: [y] }}, [expIdx]);
        }}
    }}

    function setManualFitOnPlot(show) {{
        const gd = document.getElementById('annual-chart');
        if (!gd || typeof Plotly === 'undefined' || !manualFitTraceIndices.length) return;
        const vis = manualFitTraceIndices.map(() => show);
        Plotly.restyle(gd, {{ visible: vis }}, manualFitTraceIndices);
    }}

    document.addEventListener('DOMContentLoaded', function() {{
        const gd = document.getElementById('annual-chart');
        const btnLinear = document.getElementById('scale-linear');
        const btnLog = document.getElementById('scale-log');
        if (gd && btnLinear && btnLog && typeof Plotly !== 'undefined') {{
            btnLinear.addEventListener('click', function() {{
                Plotly.relayout(gd, {{ yaxis: {{ type: 'linear', range: [0, 15000] }} }});
                btnLinear.classList.add('active');
                btnLog.classList.remove('active');
            }});
            btnLog.addEventListener('click', function() {{
                Plotly.relayout(gd, {{ yaxis: {{ type: 'log', range: [0, 5.5] }} }});
                btnLog.classList.add('active');
                btnLinear.classList.remove('active');
            }});
        }}

        ['xray', 'cryoem'].forEach(key => {{
            if (!sigmoidParams[key]) return;
            ['L', 'k', 'x0', 'b'].forEach(param => {{
                const el = document.getElementById(key + '-' + param);
                if (el) el.addEventListener('input', () => updateSigmoid(key));
            }});
        }});
        if (sigmoidParams.mfx) {{
            ['L', 'k', 'x0', 'b'].forEach(param => {{
                const el = document.getElementById('mfx-' + param);
                if (el) el.addEventListener('input', updateMfx);
            }});
        }}
        if (sigmoidParams.cryoem_exp) {{
            ['a', 'k', 'x0', 'b'].forEach(param => {{
                const el = document.getElementById('cryoem_exp-' + param);
                if (el) el.addEventListener('input', updateCryoemExp);
            }});
        }}
        const showManual = document.getElementById('show-manual-fit-on-plot');
        if (showManual) {{
            showManual.addEventListener('change', function() {{
                setManualFitOnPlot(showManual.checked);
            }});
        }}
    }});
    </script>"""

    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>PDB Statistics Dashboard</title>
    <style>
        body {{ font-family: system-ui, -apple-system, sans-serif; margin: 2rem; max-width: 1200px; color: #333; }}
        h1 {{ font-weight: 600; letter-spacing: -0.02em; }}
        .intro {{ color: #666; margin-bottom: 1.5rem; }}
        .chart-section {{ margin-bottom: 0; }}
        .annual-chart-toolbar {{
            display: flex;
            justify-content: flex-end;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.35rem;
        }}
        .scale-toggle {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.8rem;
            color: #57606a;
        }}
        .scale-toggle button {{ padding: 0.25rem 0.5rem; font-size: 0.75rem; cursor: pointer; border-radius: 4px; border: 1px solid #ccc; background: white; }}
        .scale-toggle button.active {{ background: #0969da; color: white; border-color: #0969da; }}
        .manual-fit {{
            margin: 0 0 1.5rem 0; padding: 1.25rem 1.5rem;
            background: linear-gradient(135deg, #f6f8fa 0%, #eef1f5 100%);
            border-radius: 12px; border: 1px solid #e1e4e8;
        }}
        .manual-fit h3 {{ margin: 0 0 0.25rem 0; font-size: 0.95rem; font-weight: 600; color: #24292e; }}
        .manual-fit-toolbar {{ margin: 0 0 0.5rem 0; font-size: 0.9rem; color: #24292e; }}
        .manual-fit-toggle {{ cursor: pointer; user-select: none; }}
        .manual-fit-hint {{ margin: 0 0 1rem 0; font-size: 0.85rem; color: #57606a; }}
        .param-groups {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem; }}
        .param-group {{
            min-width: 0;
            background: white; padding: 1rem 1.25rem; border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }}
        .param-group h4 {{ margin: 0 0 0.75rem 0; font-size: 0.9rem; font-weight: 600; color: #24292e; }}
        .method-label {{ font-size: 0.7em; vertical-align: middle; }}
        .sliders {{ display: flex; flex-direction: column; gap: 0.5rem; }}
        .slider-row {{
            display: grid; grid-template-columns: 1.5em 5em 3.5em 1fr;
            align-items: center; gap: 0.5rem; font-size: 0.85rem; cursor: default;
        }}
        .param-icon {{ font-size: 1rem; opacity: 0.8; text-align: center; }}
        .param-name {{ color: #57606a; }}
        .param-val {{ font-family: ui-monospace, monospace; font-size: 0.8rem; color: #0969da; min-width: 3em; }}
        .slider {{ width: 100%; height: 6px; border-radius: 3px; accent-color: #0969da; }}
        .total-section {{ margin-top: 0; }}
        .table-section {{ margin-top: 2rem; }}
    </style>
</head>
<body>
    <h1>PDB Statistics Dashboard</h1>
    <p class="intro">Interactive charts and data table. Hover for details, zoom, pan, and use the legend to toggle traces.</p>
    <div class="chart-section">
        <div class="annual-chart-toolbar">
            <div class="scale-toggle">
                <span>Y-axis scale</span>
                <button type="button" id="scale-linear" class="active">Linear</button>
                <button type="button" id="scale-log">Log</button>
            </div>
        </div>
        {annual_html}
    </div>
    {controls_html}
    <div class="total-section">{total_html}</div>
    {controls_script}
    <div class="table-section">{table_html}</div>
</body>
</html>"""

    Path(output_path).write_text(full_html, encoding="utf-8")
    print(f"Dashboard saved to {output_path}")


if __name__ == "__main__":
    build_dashboard()
