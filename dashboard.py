"""
Generate an interactive HTML dashboard for PDB statistics from the annual per-method CSV.
"""

import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy.optimize import curve_fit


def sigmoid(x: np.ndarray, L: float, k: float, x0: float, b: float) -> np.ndarray:
    """Logistic sigmoid: L / (1 + exp(-k*(x - x0))) + b"""
    return L / (1 + np.exp(-k * (x - x0))) + b


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
) -> None:
    """
    Load the PDB annual statistics CSV and build an interactive HTML dashboard
    with charts and a data table.

    Args:
        csv_path: Path to the CSV file. Defaults to the Sheet1 file in the same directory.
        output_path: Path for the output HTML file. Defaults to pdb_dashboard.html.
        milestones_path: Path to milestones CSV (Year, Milestone). Defaults to milestones.csv.
    """
    if csv_path is None:
        csv_path = Path(__file__).parent / "PDB Annual per method - Sheet1.csv"
    if output_path is None:
        output_path = Path(__file__).parent / "pdb_dashboard.html"
    if milestones_path is None:
        milestones_path = Path(__file__).parent / "milestones.csv"

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

    # Annual chart (standalone)
    fig_annual = go.Figure()
    fig_annual.add_trace(
        go.Scatter(
            x=df["Year"],
            y=df["Number of Structures Released Annually X-ray"],
            name="X-ray",
            legendgroup="X-ray",
            mode="lines+markers",
            line=dict(width=2, color=xray_color),
            marker=dict(color=xray_color),
        ),
    )
    fig_annual.add_trace(
        go.Scatter(
            x=df["Year"],
            y=df["Number of Structures Released Annually cryoEM"],
            name="cryoEM",
            legendgroup="cryoEM",
            mode="lines+markers",
            line=dict(width=2, color=cryoem_color),
            marker=dict(color=cryoem_color),
        ),
    )

    # Fit sigmoids to annual curves and extend to 2040
    years_extended = np.arange(df["Year"].min(), 2041).tolist()
    sigmoid_params = {}
    sigmoid_trace_indices = {}
    trace_idx = 2  # First sigmoid trace index (after X-ray and cryoEM data traces)
    for col, color, name, legendgroup, key in [
        ("Number of Structures Released Annually X-ray", xray_color, "X-ray (sigmoid fit)", "X-ray-sigmoid", "xray"),
        ("Number of Structures Released Annually cryoEM", cryoem_color, "cryoEM (sigmoid fit)", "cryoEM-sigmoid", "cryoem"),
    ]:
        df_fit = df[df["Year"] != 2026]
        x_data = df_fit["Year"].values.astype(float)
        y_data = df_fit[col].values.astype(float)
        try:
            p0 = (y_data.max() - y_data.min(), 0.1, x_data.mean(), y_data.min())
            popt, pcov = curve_fit(sigmoid, x_data, y_data, p0=p0, maxfev=20000)
            L, k, x0, b = popt
            years_arr = np.array(years_extended)
            y_fit = sigmoid(years_arr, *popt)
            y_fit = np.maximum(y_fit, 0)

            # Confidence band (95% CI)
            y_lo, y_hi = confidence_band(years_arr, popt, pcov)
            y_lo = np.maximum(y_lo, 0)
            y_hi = np.maximum(y_hi, 0)

            # Add confidence band: lower trace, then upper with fill to lower
            band_name = name.replace(" (sigmoid fit)", " 95% CI")
            fill_color = _hex_to_rgba(color, 0.25)
            fig_annual.add_trace(
                go.Scatter(
                    x=years_extended,
                    y=y_lo.tolist(),
                    name=band_name,
                    legendgroup=f"{legendgroup}-ci",
                    showlegend=True,
                    mode="lines",
                    line=dict(width=0),
                ),
            )
            fig_annual.add_trace(
                go.Scatter(
                    x=years_extended,
                    y=y_hi.tolist(),
                    name=band_name,
                    legendgroup=f"{legendgroup}-ci",
                    showlegend=False,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=fill_color,
                ),
            )

            # Best-fit sigmoid (static, from curve_fit)
            fig_annual.add_trace(
                go.Scatter(
                    x=years_extended,
                    y=y_fit.tolist(),
                    name=name,
                    legendgroup=legendgroup,
                    mode="lines",
                    line=dict(width=2, color=color, dash="dash"),
                ),
            )
            # Adjusted sigmoid (responds to sliders)
            adj_name = name.replace(" (sigmoid fit)", " (adjusted)")
            fig_annual.add_trace(
                go.Scatter(
                    x=years_extended,
                    y=y_fit.tolist(),
                    name=adj_name,
                    legendgroup=f"{legendgroup}-adj",
                    mode="lines",
                    line=dict(width=2, color=color),
                ),
            )
            # Store params and slider bounds
            sigmoid_params[key] = {
                "L": float(L), "k": float(k), "x0": float(x0), "b": float(b),
                "L_min": max(0, L * 0.1), "L_max": L * 2,
                "k_min": 0.001, "k_max": 1,
                "x0_min": 1975, "x0_max": 2040,
                "b_min": max(-5000, b - 3000), "b_max": min(10000, b + 3000),
            }
            sigmoid_trace_indices[key] = trace_idx + 3  # adjusted sigmoid is 4th trace per method
            trace_idx += 4  # CI lower, CI upper, best fit, adjusted

            # X-ray only: add "X-ray + MFX (adjusted)" trace (sum of X-ray adjusted + MFX effect sigmoid)
            if key == "xray":
                L_mfx, k_mfx, x0_mfx, b_mfx = 2000.0, 0.71, 2030.0, 0.0
                y_mfx = sigmoid(years_arr, L_mfx, k_mfx, x0_mfx, b_mfx)
                y_mfx = np.maximum(y_mfx, 0)
                y_xray_plus_mfx = (np.array(y_fit) + y_mfx).tolist()
                fig_annual.add_trace(
                    go.Scatter(
                        x=years_extended,
                        y=y_xray_plus_mfx,
                        name="X-ray + MFX (adjusted)",
                        legendgroup="xray-mfx-adj",
                        mode="lines",
                        line=dict(width=2, color=xray_color, dash="dot"),
                    ),
                )
                sigmoid_params["mfx"] = {
                    "L": L_mfx, "k": k_mfx, "x0": x0_mfx, "b": b_mfx,
                    "L_min": 0, "L_max": 5000,
                    "k_min": 0.001, "k_max": 1,
                    "x0_min": 2010, "x0_max": 2040,
                    "b_min": -500, "b_max": 500,
                }
                sigmoid_trace_indices["xray_plus_mfx"] = trace_idx
                trace_idx += 1
        except (RuntimeError, ValueError):
            pass

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
            )
        )

    # Annual chart layout
    fig_annual.update_layout(
        title="PDB Structures Released Annually by Method",
        xaxis_title="Year",
        yaxis_title="Structures Released Annually",
        template="plotly_white",
        height=420,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(t=80),
        shapes=shapes,
        annotations=annotations,
    )
    fig_annual.update_xaxes(range=[df["Year"].min() - 1, 2041])

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

    # Create HTML with charts + data table
    annual_html = fig_annual.to_html(full_html=False, include_plotlyjs="cdn", div_id="annual-chart")
    total_html = fig_total.to_html(full_html=False, include_plotlyjs=False, div_id="total-chart")

    table_fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(df_display.columns),
                    fill_color="paleturquoise",
                    align="left",
                ),
                cells=dict(
                    values=[df_display[col] for col in df_display.columns],
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

    # Build sigmoid controls HTML and JS
    controls_html = ""
    controls_script = ""
    if sigmoid_params:
        controls_html = '<div class="sigmoid-controls"><h3>Sigmoid fit parameters</h3><p class="sigmoid-hint">Adjust the parameters to explore scenarios; the <strong>adjusted</strong> curve updates in real time. <strong>X-ray + MFX</strong> is the sum of X-ray (adjusted) and the MFX effect sigmoid. Use the legend to toggle traces.</p><div class="param-groups">'
        for key, label in [("xray", "X-ray"), ("mfx", "MFX effect"), ("cryoem", "cryoEM")]:
            if key not in sigmoid_params:
                continue
            p = sigmoid_params[key]
            method_color = xray_color if key == "xray" else (xray_color if key == "mfx" else cryoem_color)
            controls_html += f'<div class="param-group"><h4><span class="method-label" style="color: {method_color}">●</span> {label}</h4><div class="sliders">'
            for param in ["L", "k", "x0", "b"]:
                icon, param_label, tooltip = param_info[param]
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
    const yearsExtended = {json.dumps(years_extended)};

    function sigmoid(x, L, k, x0, b) {{
        return L / (1 + Math.exp(-k * (x - x0))) + b;
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

    document.addEventListener('DOMContentLoaded', function() {{
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
        .sigmoid-controls {{
            margin: 0 0 1.5rem 0; padding: 1.25rem 1.5rem;
            background: linear-gradient(135deg, #f6f8fa 0%, #eef1f5 100%);
            border-radius: 12px; border: 1px solid #e1e4e8;
        }}
        .sigmoid-controls h3 {{ margin: 0 0 0.25rem 0; font-size: 0.95rem; font-weight: 600; color: #24292e; }}
        .sigmoid-hint {{ margin: 0 0 1rem 0; font-size: 0.85rem; color: #57606a; }}
        .param-groups {{ display: flex; flex-wrap: wrap; gap: 2rem; }}
        .param-group {{
            flex: 1; min-width: 280px;
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
    <div class="chart-section">{annual_html}</div>
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
