
from pathlib import Path
import textwrap
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Global GDP Insights", page_icon="🌍", layout="wide")
DATA_PATH = Path("data/Merged_tidy.csv")

@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"country","iso3","year","gdp_usd"}
    assert required.issubset(df.columns), f"CSV must contain {sorted(required)}"
    if "gdp_growth(%)" not in df.columns:
        st.warning("Column 'gdp_growth(%)' not found — growth metric will be disabled.")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["gdp_usd"] = pd.to_numeric(df["gdp_usd"], errors="coerce")
    if "gdp_growth(%)" in df.columns:
        df["gdp_growth(%)"] = pd.to_numeric(df["gdp_growth(%)"], errors="coerce")
    df = df.dropna(subset=["year"]).assign(year=lambda d: d["year"].astype(int))
    return df

if not DATA_PATH.exists():
    st.error("data/Merged_tidy.csv not found. Add it to the repo under data/ and redeploy.")
    st.stop()

df = load_data(DATA_PATH)

# Sidebar
st.sidebar.title("Filters")
min_year, max_year = int(df["year"].min()), int(df["year"].max())
year = st.sidebar.slider("Year", min_year, max_year, value=max_year, step=1)

countries = sorted(df["country"].dropna().unique().tolist())
default_countries = [c for c in ["United States","China","India"] if c in countries] or countries[:5]
sel_countries = st.sidebar.multiselect("Countries", countries, default=default_countries)

metrics = ["gdp_usd"] + (["gdp_growth(%)"] if "gdp_growth(%)" in df.columns else [])
metric = st.sidebar.radio(
    "Metric", metrics,
    format_func=lambda m: "GDP (current US$)" if m=="gdp_usd" else "GDP annual % growth"
)
topn = st.sidebar.slider("Top N (table)", 5, 50, 10, step=5)
st.sidebar.markdown("---"); st.sidebar.caption("Data: World Bank WDI")

# Header
st.title("🌍 Global GDP Insights Dashboard")
st.caption("Explore GDP levels and growth by country over time. (World Bank WDI)")

# KPIs
ydf = df[df["year"] == year].copy()
c1, c2, c3, c4 = st.columns(4)
if metric == "gdp_usd":
    world_total = ydf["gdp_usd"].sum(skipna=True)
    top_row = ydf.loc[ydf["gdp_usd"].idxmax()] if ydf["gdp_usd"].notna().any() else None
    median_val = ydf["gdp_usd"].median(skipna=True)
    c1.metric("World GDP", f"${world_total/1e12:.2f}T")
    c2.metric("Top economy", f"{top_row['country']} (${top_row['gdp_usd']/1e12:.2f}T)" if top_row is not None else "—")
    c3.metric("Median GDP", f"${median_val/1e9:.1f}B")
    c4.metric("Countries", f"{ydf['country'].nunique():,}")
else:
    mean_g = ydf["gdp_growth(%)"].mean(skipna=True)
    top_row = ydf.loc[ydf["gdp_growth(%)"].idxmax()] if ydf["gdp_growth(%)"].notna().any() else None
    median_val = ydf["gdp_growth(%)"].median(skipna=True)
    c1.metric("Avg growth", f"{mean_g:.2f}%")
    c2.metric("Fastest", f"{top_row['country']} ({top_row['gdp_growth(%)']:.2f}%)" if top_row is not None else "—")
    c3.metric("Median growth", f"{median_val:.2f}%")
    c4.metric("Countries", f"{ydf['country'].nunique():,}")

# Map
st.subheader("World map")
val_col = metric
map_df = ydf[ydf["iso3"].astype(str).str.len() == 3].copy()
fig_map = px.choropleth(map_df, locations="iso3", color=val_col, hover_name="country",
                        color_continuous_scale="Viridis",
                        labels={"gdp_usd":"GDP (US$)", "gdp_growth(%)":"Growth %"})
fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig_map, use_container_width=True)

# Time series
st.subheader("Trend over time (selected countries)")
if sel_countries:
    line_df = df[df["country"].isin(sel_countries)].copy()
    fig_ts = px.line(line_df, x="year", y=val_col, color="country", markers=True,
                     labels={"year":"Year", val_col:"GDP (US$)" if val_col=="gdp_usd" else "Growth %"})
    fig_ts.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_ts, use_container_width=True)
else:
    st.info("Select at least one country to see the trend.")

# Table
st.subheader(f"Top {topn} by {'GDP (US$)' if val_col=='gdp_usd' else 'Growth %'} in {year}")
rank_df = ydf.dropna(subset=[val_col]).sort_values(val_col, ascending=False).head(topn)
st.dataframe(rank_df[["country", val_col]].rename(columns={"country":"Country", val_col:"Value"}),
             use_container_width=True)

st.markdown("---")
st.markdown(textwrap.dedent(\"\"\"
**Usage**: The app reads `data/Merged_tidy.csv` with columns `country, iso3, year, gdp_usd, gdp_growth(%)`.
**Source**: World Bank WDI.
\"\""))
