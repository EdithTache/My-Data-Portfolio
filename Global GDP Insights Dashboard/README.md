# Global GDP Insights Dashboard

Interactive Streamlit app exploring GDP (current US$) and GDP growth (annual %) by country and year.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data
Expects `data/Merged_tidy.csv` with columns: country, iso3, year, gdp_usd, gdp_growth(%)
