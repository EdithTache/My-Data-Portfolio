# app.py — Global GDP Insights Dashboard (Dash)
from pathlib import Path
import os
import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc, dash_table, Input, Output, State

# ---- Config ----
DATA_PATH = Path(os.getenv("DATA_PATH", "data/tidy/Merged_tidy.csv"))

# ---- Load data once ----
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"CSV not found at {DATA_PATH}. "
        "Place Merged_tidy.csv at data/tidy/Merged_tidy.csv or set DATA_PATH env var."
    )

df = pd.read_csv(DATA_PATH)
required_cols = {"country", "iso3", "year", "gdp_usd"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# types
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
df["gdp_usd"] = pd.to_numeric(df["gdp_usd"], errors="coerce")
if "gdp_growth(%)" in df.columns:
    df["gdp_growth(%)"] = pd.to_numeric(df["gdp_growth(%)"], errors="coerce")

# choices
years = sorted(df["year"].dropna().unique().astype(int))
countries = sorted(df["country"].dropna().unique())
has_growth = "gdp_growth(%)" in df.columns
metrics = ["gdp_usd"] + (["gdp_growth(%)"] if has_growth else [])

# ---- App ----
app = Dash(__name__, title="Global GDP Insights")
server = app.server  # for gunicorn

app.layout = html.Div(
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "1rem"},
    children=[
        html.H2("🌍 Global GDP Insights Dashboard"),
        html.P([
            "Explore GDP levels and growth by country over time. ",
            html.A("Source: World Bank Data360",
                   href="https://data360.worldbank.org/en/search?search=GDP&themeAndTopics=P3",
                   target="_blank")
        ]),

        # Controls
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr 1fr", "gap": "0.75rem"},
            children=[
                html.Div([
                    html.Label("Year"),
                    dcc.Slider(
                        min=years[0], max=years[-1],
                        step=1, value=years[-1],
                        marks=None, tooltip={"placement": "bottom", "always_visible": True},
                        id="year"
                    ),
                ], style={"gridColumn": "1 / span 4"}),

                html.Div([
                    html.Label("Metric"),
                    dcc.RadioItems(
                        metrics,
                        value=metrics[0],
                        id="metric",
                        inline=True,
                        labelStyle={"marginRight": "1rem"}
                    )
                ]),

                html.Div([
                    html.Label("Countries"),
                    dcc.Dropdown(
                        options=countries,
                        value=[c for c in ["United States", "China", "India"] if c in countries] or countries[:5],
                        multi=True, id="countries"
                    )
                ], style={"gridColumn": "2 / span 2"}),

                html.Div([
                    html.Label("Top N (table)"),
                    dcc.Slider(min=5, max=50, step=5, value=10, id="topn",
                               marks={n: str(n) for n in range(5, 55, 5)})
                ])
            ]
        ),

        html.Hr(),

        # KPIs
        html.Div(id="kpi-row", style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "0.75rem"}),

        html.H3("World map"),
        dcc.Graph(id="map"),

        html.H3("Trend over time (selected countries)"),
        dcc.Graph(id="timeseries"),

        html.H3(id="table-title"),
        dash_table.DataTable(
            id="rank-table",
            style_table={"overflowX": "auto"},
            style_cell={"padding": "6px", "fontSize": 14},
            sort_action="native",
            page_size=15
        ),

        html.Hr(),
        html.Small(f"Reads: {DATA_PATH.as_posix()}  •  Columns: "
                   f"{', '.join(df.columns)}")
    ]
)

# ---- Callbacks ----

@app.callback(
    Output("kpi-row", "children"),
    Output("map", "figure"),
    Output("timeseries", "figure"),
    Output("rank-table", "data"),
    Output("rank-table", "columns"),
    Output("table-title", "children"),
    Input("year", "value"),
    Input("metric", "value"),
    Input("countries", "value"),
    Input("topn", "value"),
)
def update(year, metric, selected_countries, topn):
    # filter by year for KPIs/map/table
    ydf = df[df["year"] == year].copy()

    # KPIs
    if metric == "gdp_usd":
        world_total = ydf["gdp_usd"].sum(skipna=True)
        top_row = ydf.loc[ydf["gdp_usd"].idxmax()] if ydf["gdp_usd"].notna().any() else None
        median_val = ydf["gdp_usd"].median(skipna=True)
        kpis = [
            kpi_box("World GDP", f"${world_total/1e12:.2f}T"),
            kpi_box("Top economy",
                    f"{top_row['country']} (${top_row['gdp_usd']/1e12:.2f}T)" if top_row is not None else "—"),
            kpi_box("Median GDP", f"${median_val/1e9:.1f}B"),
            kpi_box("Countries", f"{ydf['country'].nunique():,}")
        ]
    else:
        mean_g = ydf["gdp_growth(%)"].mean(skipna=True)
        top_row = ydf.loc[ydf["gdp_growth(%)"].idxmax()] if ydf["gdp_growth(%)"].notna().any() else None
        median_val = ydf["gdp_growth(%)"].median(skipna=True)
        kpis = [
            kpi_box("Avg growth", f"{mean_g:.2f}%"),
            kpi_box("Fastest",
                    f"{top_row['country']} ({top_row['gdp_growth(%)']:.2f}%)" if top_row is not None else "—"),
            kpi_box("Median growth", f"{median_val:.2f}%"),
            kpi_box("Countries", f"{ydf['country'].nunique():,}")
        ]

    # Map
    map_df = ydf[ydf["iso3"].astype(str).str.len() == 3].copy()
    fig_map = px.choropleth(
        map_df, locations="iso3", color=metric, hover_name="country",
        color_continuous_scale="Viridis",
        labels={"gdp_usd": "GDP (US$)", "gdp_growth(%)": "Growth %"}
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=420)

    # Time series
    sel = selected_countries or []
    line_df = df[df["country"].isin(sel)].copy()
    y_label = "GDP (US$)" if metric == "gdp_usd" else "Growth %"
    fig_ts = px.line(line_df, x="year", y=metric, color="country", markers=True,
                     labels={"year": "Year", metric: y_label})
    fig_ts.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=420)

    # Table
    rank_df = ydf.dropna(subset=[metric]).sort_values(metric, ascending=False).head(int(topn))
    table_data = rank_df[["country", metric]].rename(columns={"country": "Country", metric: "Value"}).to_dict("records")
    table_cols = [{"name": c, "id": c} for c in ["Country", "Value"]]
    table_title = f"Top {topn} by {'GDP (US$)' if metric=='gdp_usd' else 'Growth %'} in {year}"

    return kpis, fig_map, fig_ts, table_data, table_cols, table_title


def kpi_box(title, value):
    return html.Div(
        style={
            "border": "1px solid #eee", "borderRadius": "8px",
            "padding": "12px", "background": "#fafafa"
        },
        children=[html.Div(title, style={"color": "#666", "fontSize": "0.9rem"}),
                  html.Div(value, style={"fontWeight": 700, "fontSize": "1.2rem"})]
    )


if __name__ == "__main__":
    # Run locally: python app.py  -> http://127.0.0.1:8050
    app.run_server(host="0.0.0.0", port=8050, debug=False)
