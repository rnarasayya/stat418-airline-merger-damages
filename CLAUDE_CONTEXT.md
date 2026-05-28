# Airline Merger Damages Estimator -- Project Context for Claude Code

## What This Document Is

This is a briefing document for Claude Code. Read this entire file before writing any code.
Everything marked COMPLETE must not be touched. The task is to modify the existing
Streamlit app as described in the Change Requests section below.

---

## Project Overview

**Project:** Airline Merger Damages Estimator
**Course:** STAT 418 -- Data/ML/AI Applications (UCLA)
**Goal:** A deployed web application estimating consumer harm from the American Airlines /
US Airways merger (closed December 2013) using causal inference methods.

**Core finding:** The merger caused a statistically significant 2.57% fare increase on routes
where AA and US Airways previously competed. Total consumer harm: $3.97 billion across 622M passengers.

---

## Confirmed Modeling Results -- Do Not Re-Run

| Model | ATE (log pts) | ATE (% fare change) |
|---|---|---|
| Pooled OLS | -0.0277 | -2.73% |
| Two-Way FE | -0.0257 | -2.54% |
| Double ML (PRIMARY) | +0.0254 | +2.57% |
| Causal Forest (mean ITE) | +0.0438 | +4.48% |

Double ML 95% CI: [+1.13%, +5.44%] -- statistically significant
Total damages: $3.97B | Affected passengers: 622.2M | Avg overcharge per ticket: $6.73

---

## Flask API -- COMPLETE, DO NOT TOUCH api/

Running locally at http://127.0.0.1:8080. All four endpoints confirmed working:

- GET /health -- returns status ok, ate_pct 2.57, routes_available 1214
- GET /routes -- returns 200 treated routes sorted by damages descending
- GET /damages -- returns total_damages_bn, top_routes list, model_comparison table
- POST /predict -- body: {"route": "LAX-JFK"}, returns avg_fare_actual, avg_fare_but_for,
  avg_overcharge, overcharge_pct, total_damages, total_passengers, found_in_damages bool

---

## Current Status

### COMPLETE -- Do Not Touch
- [x] Data download, all notebooks, model artifacts
- [x] Flask API (api/app.py, api/requirements.txt, api/Dockerfile)
- [x] API tests (tests/test_api.py)
- [x] Streamlit app (streamlit_app/app.py) -- working locally, needs two modifications below

### MODIFY NOW
- [ ] Two changes to streamlit_app/app.py -- see Change Requests below
- [ ] Update streamlit_app/Dockerfile to copy panel.parquet

---

## Panel Data Reference

`data/processed/panel.parquet` -- 122,787 rows, route-quarter level.

Key columns needed for the route history chart:
- `route`: str, e.g. "LAX-JFK"
- `Year`: int
- `Quarter`: int (1-4)
- `quarter_idx`: int, 0=2010Q1 to 31=2017Q4
- `avg_fare`: float, mean fare in dollars
- `treated`: int, 1=treated route
- `post`: int, 1=Q4 2013 or later

The merger date is quarter_idx=15 (Q4 2013).
The panel file is at: data/processed/panel.parquet (relative to project root)
In Docker it will be copied to: /app/data/panel.parquet

---

## Project Directory Structure

```
airline-merger-damages/
|
+-- CLAUDE_CONTEXT.md
+-- data/
|   +-- processed/
|       +-- panel.parquet      <- needed by Streamlit for history chart
+-- api/                       <- COMPLETE, do not modify
+-- streamlit_app/             <- MODIFY app.py and Dockerfile
|   +-- app.py                 <- modify in place
|   +-- requirements.txt       <- no changes needed
|   +-- Dockerfile             <- add panel.parquet copy step
+-- models/artifacts/          <- COMPLETE, do not modify
+-- notebooks/                 <- ALL COMPLETE, do not modify
+-- tests/                     <- COMPLETE, do not modify
```

---

## CHANGE REQUESTS -- These Are the Only Two Things to Build

### Change 1: Move the All Treated Routes Table to Executive Summary Page

**What to do:**
The ranked table of all treated routes currently lives at the bottom of the Route Analysis
page. Move it to the Executive Summary page, placing it below the model comparison table.
Remove it from the Route Analysis page entirely.

**On Executive Summary page, the order should be:**
1. Title and subtitle (unchanged)
2. Four metric cards (unchanged)
3. Key finding box (unchanged)
4. Top 20 routes bar chart (unchanged)
5. Model comparison table (unchanged)
6. Sign flip methodology note (unchanged)
7. NEW: Section header "All Treated Routes -- Ranked by Consumer Harm"
8. NEW: The full ranked table (move from Route Analysis, same code, same columns)

**Table code to move (currently at bottom of Route Analysis page):**
```python
st.markdown('<div class="section-header">All Treated Routes -- Ranked by Consumer Harm</div>', unsafe_allow_html=True)
routes_df = pd.DataFrame(routes_data["route_details"])
routes_df = routes_df.rename(columns={
    "route":            "Route",
    "avg_fare_actual":  "Avg Fare ($)",
    "avg_fare_but_for": "Counterfactual Fare ($)",
    "avg_overcharge":   "Overcharge ($)",
    "total_damages":    "Total Harm ($)",
    "total_passengers": "Passengers",
})
routes_df["Total Harm ($M)"]  = (routes_df["Total Harm ($)"] / 1e6).round(1)
routes_df["Passengers (M)"]   = (routes_df["Passengers"] / 1e6).round(2)
display_cols = [
    "Route", "Avg Fare ($)", "Counterfactual Fare ($)",
    "Overcharge ($)", "Total Harm ($M)", "Passengers (M)"
]
st.dataframe(
    routes_df[display_cols],
    use_container_width=True,
    hide_index=True,
    height=400
)
```

NOTE: The table needs routes_data from fetch_routes(). On the Executive Summary page,
routes_data is not currently fetched. Add this fetch at the top of the Executive Summary
page block, alongside the existing fetch_damages call:

```python
with st.spinner("Loading analysis..."):
    data = fetch_damages(top_n=20)
    routes_data = fetch_routes()
```

---

### Change 2: Add Route Price History Line Chart to Route Analysis Page

**What to do:**
Load panel.parquet once at app startup using st.cache_data. When a user selects a route
on the Route Analysis page, show a line chart of that route's average quarterly fare from
2010 Q1 to 2017 Q4, with a vertical line marking the merger date (Q4 2013).

**Panel data loading -- add near the top of app.py, after imports and API_URL:**

```python
PANEL_PATH = os.environ.get("PANEL_PATH", "../data/processed/panel.parquet")

@st.cache_data
def load_panel():
    try:
        df = pd.read_parquet(PANEL_PATH, columns=["route", "Year", "Quarter",
                                                    "quarter_idx", "avg_fare",
                                                    "treated", "post"])
        return df
    except Exception as e:
        return None
```

**Route history chart -- add this to the Route Analysis page, AFTER the existing
two-column metrics and fare comparison bar chart, and BEFORE where the ranked table
used to be (which is now removed per Change 1):**

```python
# Route price history chart
st.markdown('<div class="section-header">Fare History: 2010-2017</div>', unsafe_allow_html=True)

panel = load_panel()
if panel is not None:
    route_history = panel[panel["route"] == selected_route].sort_values("quarter_idx")

    if not route_history.empty:
        # Build readable x-axis labels: "2010 Q1", "2010 Q2", etc.
        route_history = route_history.copy()
        route_history["period"] = route_history["Year"].astype(str) + " Q" + route_history["Quarter"].astype(str)

        fig_hist = go.Figure()

        # Pre-merger line (navy)
        pre = route_history[route_history["post"] == 0]
        post = route_history[route_history["post"] == 1]

        fig_hist.add_trace(go.Scatter(
            x=pre["period"],
            y=pre["avg_fare"],
            mode="lines+markers",
            name="Pre-merger",
            line=dict(color="#1a2744", width=2),
            marker=dict(size=5),
        ))

        # Post-merger line (red)
        fig_hist.add_trace(go.Scatter(
            x=post["period"],
            y=post["avg_fare"],
            mode="lines+markers",
            name="Post-merger",
            line=dict(color="#c0392b", width=2),
            marker=dict(size=5),
        ))

        # Add vertical line at merger date using a shape
        # Find the index position of the merger quarter for the vline
        merger_period = "2013 Q4"
        all_periods = route_history["period"].tolist()
        if merger_period in all_periods:
            merger_x = all_periods.index(merger_period)
        else:
            merger_x = 15  # fallback

        fig_hist.add_vline(
            x=merger_period,
            line_dash="dash",
            line_color="#7f8c8d",
            line_width=1.5,
            annotation_text="Merger (Dec 2013)",
            annotation_position="top right",
            annotation_font_size=11,
            annotation_font_color="#7f8c8d",
        )

        fig_hist.update_layout(
            title=f"Average Fare History: {selected_route}",
            xaxis_title="Quarter",
            yaxis_title="Average Fare ($)",
            height=380,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(
                showgrid=True,
                gridcolor="#f0f0f0",
                tickangle=45,
                # Show only annual ticks to avoid crowding
                tickvals=[f"{y} Q1" for y in range(2010, 2018)],
                ticktext=[str(y) for y in range(2010, 2018)],
            ),
            yaxis=dict(showgrid=True, gridcolor="#f0f0f0",
                       tickprefix="$", tickformat=",.0f"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            margin=dict(l=60, r=40, t=60, b=80),
            font=dict(family="Arial", size=12),
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info(f"No fare history available for route {selected_route}.")
else:
    st.warning("Panel data not available -- fare history chart cannot be displayed.")
```

---

### Change 3: Update Dockerfile to Include Panel Data

The Streamlit Dockerfile needs to copy panel.parquet into the container and set
the PANEL_PATH env var. The build context is the project root.

Updated Dockerfile:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY streamlit_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streamlit_app/app.py .

# Copy panel data for route history chart
COPY data/processed/panel.parquet ./data/panel.parquet

ENV API_URL=https://your-api-url.run.app
ENV PANEL_PATH=/app/data/panel.parquet

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

---

## Important Notes for Claude Code

1. Modify streamlit_app/app.py in place -- do not rewrite the whole file from scratch.
   Only make the targeted changes described in Change 1 and Change 2.
2. The Methodology page is working correctly -- do not touch it.
3. No emoji anywhere -- Windows cp1252 encoding
4. Use st.cache_data for load_panel()
5. PANEL_PATH must come from os.environ.get() so it works both locally and in Docker
6. Locally panel.parquet is at ../data/processed/panel.parquet relative to streamlit_app/
7. Use plotly not matplotlib
8. "Counterfactual Fare" not "But-For Fare" throughout

---

## Technical Constraints -- Never Change

- Python: 3.11 (Windows environment)
- No emoji or special unicode anywhere
- App framework: Streamlit (not Shiny)
- Deployment: Google Cloud Run
- Charts: plotly (not matplotlib)

---

## Presentation Deadlines

- June 1, 2026 -- Final slides due
- June 2, 2026 -- GitHub repo + deployed app + API live
- June 9, 2026 -- Services must still be running for grading

---

## Claude Code Prompt -- Run This Now

> "Read CLAUDE_CONTEXT.md in full. The Streamlit app at streamlit_app/app.py is already built and working -- do not rewrite it from scratch. Make exactly the two targeted modifications described in the Change Requests section: (1) move the ranked routes table from the Route Analysis page to the bottom of the Executive Summary page, and (2) add a route fare history line chart to the Route Analysis page using panel.parquet loaded directly in Streamlit. Also update streamlit_app/Dockerfile to copy panel.parquet as described in Change 3. Do not touch the Methodology page, api/, notebooks/, or models/. No emoji or special unicode anywhere."
