import os
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="AA/US Airways Merger Damages Estimator",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8080")
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


st.markdown("""
<style>
    .main-header {
        font-size: 2rem; font-weight: 700;
        color: #1a2744;
        border-bottom: 3px solid #c0392b;
        padding-bottom: 0.5rem; margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.2rem; font-weight: 600;
        color: #1a2744; margin-top: 1.5rem; margin-bottom: 0.5rem;
    }
    .finding-box {
        background: #fff8f8; border: 1px solid #c0392b;
        border-radius: 6px; padding: 1rem 1.5rem; margin: 1rem 0;
    }
    .methodology-note {
        background: #f0f4ff; border-left: 4px solid #1a2744;
        padding: 0.75rem 1rem; border-radius: 4px;
        font-size: 0.9rem; color: #444;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def fetch_damages(top_n=20):
    try:
        r = requests.get(f"{API_URL}/damages", params={"top_n": top_n}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API connection failed: {e}")
        return None


@st.cache_data(ttl=300)
def fetch_routes():
    try:
        r = requests.get(f"{API_URL}/routes", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API connection failed: {e}")
        return None


def predict_route(route):
    try:
        r = requests.post(f"{API_URL}/predict", json={"route": route}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None


st.sidebar.markdown("## Navigation")
page = st.sidebar.radio(
    "",
    ["Executive Summary", "Route Analysis", "Methodology"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Study Details**")
st.sidebar.markdown("- Merger: AA / US Airways")
st.sidebar.markdown("- Closed: December 2013")
st.sidebar.markdown("- Study period: 2010-2017")
st.sidebar.markdown("- Method: Double ML (LightGBM)")
st.sidebar.markdown("- Data: DOT DB1B (10% ticket sample)")

try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    st.sidebar.markdown("---")
    st.sidebar.success("API connected")
except Exception:
    st.sidebar.error("API offline -- start Flask API on port 8080")


if page == "Executive Summary":
    st.markdown('<div class="main-header">American Airlines / US Airways Merger: Consumer Harm Analysis</div>', unsafe_allow_html=True)
    st.markdown("**Causal Inference Analysis of Antitrust Damages, 2014-2017** | STAT 418 Final Project")

    with st.spinner("Loading analysis..."):
        data = fetch_damages(top_n=20)
        routes_data = fetch_routes()

    if data:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Price Overcharge (Double ML)", "2.57%",
            help="Average treatment effect on log fares, 95% CI [1.13%, 5.44%]"
        )
        col2.metric(
            "Total Consumer Harm", "$3.97 Billion",
            help="Cumulative overcharge across all treated routes, 2014-2017"
        )
        col3.metric(
            "Affected Passengers", "622 Million",
            help="Total passengers on treated routes post-merger (DB1B x10 scaled)"
        )
        col4.metric(
            "Avg Overcharge per Ticket", "$6.73",
            help="Mean per-passenger overcharge on treated routes"
        )

        st.markdown("""
        <div class="finding-box">
        <strong>Key Finding:</strong> The merger caused a statistically significant
        <strong>2.57% fare increase</strong> on routes where AA and US Airways previously
        competed directly. Standard linear regression methods produce a spurious negative
        estimate due to the confounding effect of the 2014-2015 oil price crash. Double ML
        correctly absorbs this nonlinearity via LightGBM residualization.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Top 20 Routes by Estimated Consumer Harm (2014-2017)</div>', unsafe_allow_html=True)
        top_df = pd.DataFrame(data["top_routes"])
        top_df["damages_M"] = top_df["total_damages"] / 1e6

        fig = go.Figure(go.Bar(
            x=top_df["damages_M"][::-1],
            y=top_df["route"][::-1],
            orientation="h",
            marker_color="#c0392b",
            text=[f"${v:.1f}M" for v in top_df["damages_M"][::-1]],
            textposition="outside",
        ))
        fig.update_layout(
            xaxis_title="Total Consumer Harm ($M)",
            yaxis_title="",
            height=550,
            margin=dict(l=80, r=120, t=20, b=40),
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
            yaxis=dict(showgrid=False),
            font=dict(family="Arial", size=12),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-header">Model Comparison: Why Linear Methods Fail</div>', unsafe_allow_html=True)
        comp_df = pd.DataFrame(data["model_comparison"])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        st.markdown("""
        <div class="methodology-note">
        <strong>Note on sign flip:</strong> OLS and Two-Way Fixed Effects estimate a spurious fare
        <em>decrease</em> post-merger because they cannot flexibly model the nonlinear relationship
        between jet fuel prices and fares. The 2014-2015 oil price crash coincided with the post-merger
        period, creating a confound that Double ML correctly absorbs via LightGBM residualization.
        </div>
        """, unsafe_allow_html=True)

        if routes_data:
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
            routes_df["Total Harm ($M)"] = (routes_df["Total Harm ($)"] / 1e6).round(1)
            routes_df["Passengers (M)"]  = (routes_df["Passengers"] / 1e6).round(2)
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


elif page == "Route Analysis":
    st.markdown('<div class="main-header">Route-Level Consumer Harm Estimator</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="methodology-note">
    Select a treated route (one where both AA and US Airways operated pre-merger) to see
    estimated consumer harm for that city pair based on the Double ML causal estimate.
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Loading routes..."):
        routes_data = fetch_routes()

    if routes_data:
        route_options = routes_data["routes"]

        selected_route = st.selectbox(
            "Select Route",
            options=route_options,
            index=route_options.index("LAX-JFK") if "LAX-JFK" in route_options else 0,
        )

        with st.spinner(f"Fetching estimate for {selected_route}..."):
            result = predict_route(selected_route)

        if result and result.get("found_in_damages"):
            st.markdown(f'<div class="section-header">{selected_route} -- Route Analysis</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Actual Average Fare (2014-2017)",
                    f"${result['avg_fare_actual']:.2f}"
                )
                st.metric(
                    "Counterfactual Fare (Without Merger)",
                    f"${result['avg_fare_but_for']:.2f}",
                    help="Estimated fare absent the merger, derived from Double ML ATE"
                )
                st.metric(
                    "Estimated Overcharge per Ticket",
                    f"${result['avg_overcharge']:.2f}",
                )

            with col2:
                st.metric(
                    "Total Consumer Harm",
                    f"${result['total_damages']/1e6:.1f}M",
                    help="Total overcharge x total passengers, 2014-2017"
                )
                st.metric(
                    "Affected Passengers",
                    f"{result['total_passengers']/1e6:.2f}M"
                )

                fig2 = go.Figure()
                fig2.add_trace(go.Bar(
                    x=["Actual Fare", "Counterfactual Fare\n(Without Merger)"],
                    y=[result["avg_fare_actual"], result["avg_fare_but_for"]],
                    marker_color=["#c0392b", "#1a2744"],
                    text=[f"${result['avg_fare_actual']:.2f}",
                          f"${result['avg_fare_but_for']:.2f}"],
                    textposition="outside",
                ))
                fig2.update_layout(
                    title=f"Fare Comparison: {selected_route}",
                    yaxis_title="Average Fare ($)",
                    height=320,
                    showlegend=False,
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    yaxis=dict(
                        range=[0, result["avg_fare_actual"] * 1.2],
                        showgrid=True, gridcolor="#f0f0f0"
                    ),
                    margin=dict(l=40, r=40, t=50, b=40),
                )
                st.plotly_chart(fig2, use_container_width=True)

        st.markdown('<div class="section-header">Fare History: 2010-2017</div>', unsafe_allow_html=True)

        panel = load_panel()
        if panel is not None:
            route_history = panel[panel["route"] == selected_route].sort_values("quarter_idx")

            if not route_history.empty:
                route_history = route_history.copy()
                route_history["period"] = route_history["Year"].astype(str) + " Q" + route_history["Quarter"].astype(str)

                fig_hist = go.Figure()

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

                ci_upper = post["avg_fare"] / (1 + 0.0113)
                ci_lower = post["avg_fare"] / (1 + 0.0544)

                fig_hist.add_trace(go.Scatter(
                    x=post["period"],
                    y=ci_upper,
                    mode="lines",
                    name="95% CI counterfactual range",
                    showlegend=False,
                    line=dict(color="rgba(0,0,0,0)"),
                ))

                fig_hist.add_trace(go.Scatter(
                    x=post["period"],
                    y=ci_lower,
                    mode="lines",
                    name="95% CI counterfactual range",
                    fill="tonexty",
                    fillcolor="rgba(192,57,43,0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                ))

                fig_hist.add_trace(go.Scatter(
                    x=post["period"],
                    y=post["avg_fare"],
                    mode="lines+markers",
                    name="Post-merger",
                    line=dict(color="#c0392b", width=2),
                    marker=dict(size=5),
                ))

                merger_period = "2013 Q4"
                all_periods = route_history["period"].tolist()
                if merger_period in all_periods:
                    merger_x = all_periods.index(merger_period)
                else:
                    merger_x = 15

                if merger_period in all_periods:
                    merger_idx = all_periods.index(merger_period)
                    fig_hist.add_shape(
                        type="line",
                        x0=merger_idx, x1=merger_idx,
                        y0=0, y1=1,
                        xref="x", yref="paper",
                        line=dict(color="#7f8c8d", width=1.5, dash="dash"),
                    )
                    fig_hist.add_annotation(
                        x=merger_idx,
                        y=1.05,
                        xref="x", yref="paper",
                        text="Merger (Dec 2013)",
                        showarrow=False,
                        font=dict(size=11, color="#7f8c8d"),
                        xanchor="left",
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


elif page == "Methodology":
    st.markdown('<div class="main-header">Causal Identification Methodology</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">A. Identification Strategy</div>', unsafe_allow_html=True)
    st.markdown("""
    This analysis exploits the 2013 AA/US Airways merger as a natural experiment, comparing
    routes where the two carriers previously competed directly against routes where only one
    operated.

    - **Treated routes (1,223):** Routes where both AA and US Airways operated pre-merger.
      These faced direct competitive harm from consolidation.
    - **Control routes (3,721):** Routes where only one carrier operated pre-merger.
      These serve as the counterfactual price trend.
    - **Key assumption:** Treated and control routes would have followed parallel price
      trends absent the merger. This is visually confirmed in the pre-merger period.
    """)

    st.markdown('<div class="section-header">B. Why Double ML?</div>', unsafe_allow_html=True)
    st.markdown("""
    Standard linear estimators (OLS, Two-Way Fixed Effects) produce a spurious **negative**
    treatment effect. Time fixed effects absorb economy-wide shocks uniformly -- but the
    2014-2015 jet fuel price crash affected routes differently depending on distance,
    competition level, and carrier mix. A linear model cannot capture those interactions,
    causing the fuel-driven fare decrease to leak into the treatment estimate.

    **Double ML** (Chernozhukov et al., 2018) solves this by using LightGBM gradient
    boosting to partial out all controls flexibly before estimating the causal effect,
    leaving only the price variation attributable to the merger itself.

    The three-step procedure:
    1. Residualize log fare on all controls using LightGBM
    2. Residualize the merger treatment indicator on all controls using LightGBM
    3. Regress outcome residuals on treatment residuals to recover the causal ATE
    """)

    with st.spinner("Loading model comparison..."):
        data = fetch_damages(top_n=1)
    if data:
        comp_df = pd.DataFrame(data["model_comparison"])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.markdown("""
        <div class="methodology-note">
        The sign flip from negative (OLS/TWFE) to positive (Double ML) is the central
        methodological finding. It demonstrates why flexible causal ML methods are necessary
        when nonlinear confounders coincide with the treatment period.
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">C. Damages Calculation</div>', unsafe_allow_html=True)
    st.markdown("""
    Consumer damages are quantified by comparing actual post-merger fares to the counterfactual
    fares that would have prevailed absent the merger:

    - **Counterfactual fare** = Actual fare / (1 + 0.0257)
    - **Overcharge per ticket** = Actual fare - Counterfactual fare
    - **(Actual fare - Counterfactual fare) x Passengers = Total consumer harm**
    - **95% CI on ATE:** [+1.13%, +5.44%] -- statistically significant at 5% level

    Total estimated consumer harm: **$3.97 billion** across **622 million passengers**, 2014-2017.
    """)

    st.markdown('<div class="section-header">D. Data Sources</div>', unsafe_allow_html=True)
    st.markdown("""
    | Source | Description | Coverage |
    |---|---|---|
    | DOT DB1B (BTS) | 10% random sample of domestic airline passenger tickets | 2010-2017, 32 quarters |
    | EIA Jet Fuel (FRED: WJFUELUSGULF) | Weekly US Gulf Coast jet fuel prices | 2009-2018 |
    | BLS Airline CPI (FRED: CUSR0000SETG01) | Monthly US airline fare price index | 2009-2018 |

    Final analysis dataset: **122,787 route-quarter observations** after cleaning
    (domestic nonstop, no bulk fares, fare range $20-$2,000).
    """)
