from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import altair as alt
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine

st.set_page_config(page_title="AI Brand Monitor Dashboard", layout="wide")
st.title("AI Brand Monitor Dashboard")

engine = create_engine(st.secrets["db"]["url"])


@st.cache_data(ttl=300)
def run_query(sql: str, params: Optional[Dict] = None) -> pd.DataFrame:
    """Generic helper to execute SQL queries with caching."""
    return pd.read_sql(sql, engine, params=params)


# TODO: populate with {"Friendly Name": "customer-uuid"}
CUSTOMERS: Dict[str, str] = {}


@dataclass(frozen=True)
class FilterState:
    customer_id: Optional[str]
    date_range: Optional[Tuple[date, date]]
    llms: Tuple[str, ...] = ()
    models: Tuple[str, ...] = ()
    clusters: Tuple[str, ...] = ()
    intents: Tuple[str, ...] = ()
    tones: Tuple[str, ...] = ()


def get_customer_options() -> Dict[str, str]:
    if CUSTOMERS:
        return CUSTOMERS
    df = fetch_customer_ids()
    if df.empty:
        return {}
    return {row.customer_id: row.customer_id for row in df.itertuples()}


@st.cache_data(ttl=300)
def fetch_customer_ids() -> pd.DataFrame:
    sql = """
        SELECT DISTINCT customer_id
        FROM (
            SELECT customer_id FROM v_brand_mentions_flat
            UNION
            SELECT customer_id FROM v_source_mentions_flat
            UNION
            SELECT customer_id FROM v_ai_responses_flat
        ) t
        WHERE customer_id IS NOT NULL
        ORDER BY customer_id
    """
    return run_query(sql)


@st.cache_data(ttl=300)
def fetch_date_bounds(customer_id: str) -> Tuple[Optional[date], Optional[date]]:
    sql = """
        SELECT MIN(date) AS min_date, MAX(date) AS max_date
        FROM (
            SELECT date FROM v_brand_mentions_flat WHERE customer_id = %(customer_id)s
            UNION ALL
            SELECT date FROM v_source_mentions_flat WHERE customer_id = %(customer_id)s
            UNION ALL
            SELECT date FROM v_ai_responses_flat WHERE customer_id = %(customer_id)s
        ) d
    """
    df = run_query(sql, {"customer_id": customer_id})
    if df.empty:
        return None, None
    row = df.iloc[0]
    min_date = row["min_date"]
    max_date = row["max_date"]
    min_date = pd.to_datetime(min_date).date() if pd.notnull(min_date) else None
    max_date = pd.to_datetime(max_date).date() if pd.notnull(max_date) else None
    return min_date, max_date


@st.cache_data(ttl=300)
def fetch_distinct_column_values(customer_id: str, column: str) -> List[str]:
    allowed = {"llm", "model", "cluster", "intent", "tone"}
    if column not in allowed or not customer_id:
        return []
    sql = f"""
        SELECT DISTINCT {column}
        FROM v_brand_mentions_flat
        WHERE customer_id = %(customer_id)s
          AND {column} IS NOT NULL
        ORDER BY {column}
    """
    df = run_query(sql, {"customer_id": customer_id})
    return df[column].dropna().tolist()


def build_where_clause(filters: FilterState, alias: str | None = None) -> Tuple[str, Dict]:
    prefix = f"{alias}." if alias else ""
    conditions: List[str] = []
    params: Dict = {}

    if filters.customer_id:
        conditions.append(f"{prefix}customer_id = %(customer_id)s")
        params["customer_id"] = filters.customer_id
    if filters.date_range:
        start_date, end_date = filters.date_range
        conditions.append(f"{prefix}date BETWEEN %(start_date)s AND %(end_date)s")
        params["start_date"] = start_date
        params["end_date"] = end_date

    multiselect_mapping = [
        ("llms", "llm"),
        ("models", "model"),
        ("clusters", "cluster"),
        ("intents", "intent"),
        ("tones", "tone"),
    ]
    for attr, column in multiselect_mapping:
        values = getattr(filters, attr)
        if values:
            param_name = f"{attr}_filter"
            conditions.append(f"{prefix}{column} = ANY(%({param_name})s)")
            params[param_name] = list(values)

    if not conditions:
        conditions.append("TRUE")
    return " AND ".join(conditions), params


@st.cache_data(ttl=300)
def fetch_brand_list(filters: FilterState) -> List[str]:
    where_clause, params = build_where_clause(filters)
    sql = f"""
        SELECT DISTINCT brand
        FROM v_brand_mentions_flat
        WHERE {where_clause}
          AND brand IS NOT NULL
        ORDER BY brand
    """
    df = run_query(sql, params)
    return df["brand"].dropna().tolist()


@st.cache_data(ttl=300)
def fetch_brand_metrics(filters: FilterState, brand: str) -> Dict:
    where_clause, params = build_where_clause(filters)
    params = params.copy()
    params["brand"] = brand
    sql = f"""
        SELECT
            COUNT(*) AS total_mentions,
            AVG(position) AS avg_position,
            SUM(CASE WHEN position <= 3 THEN 1 ELSE 0 END) AS top3_mentions,
            AVG(CASE WHEN position <= 3 THEN position END) AS avg_position_top3,
            COUNT(DISTINCT ai_question) AS ai_questions
        FROM v_brand_mentions_flat
        WHERE {where_clause}
          AND brand = %(brand)s
    """
    df = run_query(sql, params)
    return df.iloc[0].to_dict() if not df.empty else {}


@st.cache_data(ttl=300)
def fetch_mentions_by_llm(filters: FilterState, brand: str) -> pd.DataFrame:
    where_clause, params = build_where_clause(filters)
    params = params.copy()
    params["brand"] = brand
    sql = f"""
        SELECT COALESCE(llm, 'Unknown') AS llm, COUNT(*) AS mentions
        FROM v_brand_mentions_flat
        WHERE {where_clause}
          AND brand = %(brand)s
        GROUP BY COALESCE(llm, 'Unknown')
        ORDER BY mentions DESC
    """
    return run_query(sql, params)


@st.cache_data(ttl=300)
def fetch_brand_timeline(filters: FilterState, brand: str) -> pd.DataFrame:
    where_clause, params = build_where_clause(filters)
    params = params.copy()
    params["brand"] = brand
    sql = f"""
        SELECT date, COALESCE(llm, 'Unknown') AS llm, AVG(position) AS avg_position
        FROM v_brand_mentions_flat
        WHERE {where_clause}
          AND brand = %(brand)s
          AND position IS NOT NULL
        GROUP BY date, COALESCE(llm, 'Unknown')
        ORDER BY date
    """
    return run_query(sql, params)


@st.cache_data(ttl=300)
def fetch_top_brands(filters: FilterState, limit: int = 15) -> pd.DataFrame:
    where_clause, params = build_where_clause(filters)
    sql = f"""
        SELECT brand, COUNT(*) AS mentions, AVG(position) AS avg_position
        FROM v_brand_mentions_flat
        WHERE {where_clause}
          AND brand IS NOT NULL
        GROUP BY brand
        ORDER BY mentions DESC
        LIMIT {limit}
    """
    return run_query(sql, params)


@st.cache_data(ttl=300)
def fetch_brand_ranking(filters: FilterState, limit: int = 50) -> pd.DataFrame:
    where_clause, params = build_where_clause(filters)
    sql = f"""
        SELECT brand, AVG(position) AS avg_position, COUNT(*) AS mentions
        FROM v_brand_mentions_flat
        WHERE {where_clause}
          AND brand IS NOT NULL
        GROUP BY brand
        HAVING COUNT(*) > 0
        ORDER BY avg_position ASC NULLS LAST, mentions DESC
        LIMIT {limit}
    """
    return run_query(sql, params)


@st.cache_data(ttl=300)
def fetch_cluster_ranking(filters: FilterState) -> pd.DataFrame:
    where_clause, params = build_where_clause(filters)
    sql = f"""
        SELECT cluster, intent, COUNT(*) AS mentions, AVG(position) AS avg_position
        FROM v_brand_mentions_flat
        WHERE {where_clause}
        GROUP BY cluster, intent
        ORDER BY mentions DESC
    """
    return run_query(sql, params)


@st.cache_data(ttl=300)
def fetch_question_ranking(filters: FilterState, limit: int = 50) -> pd.DataFrame:
    where_clause, params = build_where_clause(filters)
    sql = f"""
        SELECT ai_question, COUNT(*) AS mentions, AVG(position) AS avg_position
        FROM v_brand_mentions_flat
        WHERE {where_clause}
        GROUP BY ai_question
        ORDER BY mentions DESC
        LIMIT {limit}
    """
    return run_query(sql, params)


@st.cache_data(ttl=300)
def fetch_source_mentions(filters: FilterState) -> pd.DataFrame:
    where_clause, params = build_where_clause(filters)
    sql = f"""
        SELECT date, ai_question, keyword, cluster, subcluster, volume, llm, model, url, intent, tone
        FROM v_source_mentions_flat
        WHERE {where_clause}
          AND url IS NOT NULL
    """
    return run_query(sql, params)


def extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path.split("/")[0]
        return domain.lower().strip()
    except Exception:
        return ""


def safe_int(value: Optional[float]) -> int:
    if value is None or pd.isna(value):
        return 0
    return int(value)


def format_avg(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.2f}"


def render_brand_analysis_tab(filters: FilterState) -> None:
    st.markdown("## Brand Analysis")
    if not filters.customer_id:
        st.info("Select a customer to explore brand performance.")
        return

    brand_options = fetch_brand_list(filters)
    if not brand_options:
        st.warning("No brands available for the selected filters.")
        return

    selected_brand = st.selectbox("Select a brand", brand_options)
    if not selected_brand:
        st.info("Choose a brand to see detailed metrics.")
        return

    metrics = fetch_brand_metrics(filters, selected_brand)
    metric_cols = st.columns(5)
    metric_cols[0].metric("Brand Mentions", safe_int(metrics.get("total_mentions")))
    metric_cols[1].metric("Avg Brand Position", format_avg(metrics.get("avg_position")))
    metric_cols[2].metric("Top 3 Mentions", safe_int(metrics.get("top3_mentions")))
    metric_cols[3].metric("Avg Position Top 3", format_avg(metrics.get("avg_position_top3")))
    metric_cols[4].metric("AI Questions Mentions", safe_int(metrics.get("ai_questions")))

    st.divider()
    st.markdown("### Total Mentions per LLM")
    llm_df = fetch_mentions_by_llm(filters, selected_brand)
    if llm_df.empty:
        st.info("No mention data available for this brand.")
    else:
        chart = (
            alt.Chart(llm_df)
            .mark_bar()
            .encode(
                x=alt.X("llm:N", title="LLM"),
                y=alt.Y("mentions:Q", title="Mentions"),
                tooltip=["llm", "mentions"],
            )
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("### Timeline | Avg Brand Position")
    timeline_df = fetch_brand_timeline(filters, selected_brand)
    if timeline_df.empty:
        st.info("No timeline data available.")
    else:
        chart = (
            alt.Chart(timeline_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("avg_position:Q", title="Avg Position"),
                color=alt.Color("llm:N", title="LLM"),
                tooltip=["date", "llm", alt.Tooltip("avg_position:Q", format=".2f")],
            )
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("### Top 15 Brands (Bubble Chart)")
    top_brands_df = fetch_top_brands(filters)
    if top_brands_df.empty:
        st.info("No brand aggregation available.")
    else:
        bubble_chart = (
            alt.Chart(top_brands_df)
            .mark_circle()
            .encode(
                x=alt.X("mentions:Q", title="Mentions"),
                y=alt.Y("avg_position:Q", title="Avg Position", sort="descending"),
                size=alt.Size("mentions:Q", title="Mentions"),
                tooltip=["brand", alt.Tooltip("avg_position:Q", format=".2f"), "mentions"],
                color=alt.Color("brand:N", legend=None),
            )
        )
        st.altair_chart(bubble_chart, use_container_width=True)

    st.markdown("### Brand Ranking")
    brand_ranking_df = fetch_brand_ranking(filters)
    if brand_ranking_df.empty:
        st.info("No ranking data available.")
    else:
        df = brand_ranking_df.copy()
        df.insert(0, "Rank", range(1, len(df) + 1))
        df["avg_position"] = df["avg_position"].round(2)
        df.rename(columns={"brand": "Brand", "avg_position": "Avg Position", "mentions": "Mentions"}, inplace=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Cluster Ranking")
    cluster_df = fetch_cluster_ranking(filters)
    if cluster_df.empty:
        st.info("No cluster data available.")
    else:
        df = cluster_df.copy()
        df["avg_position"] = df["avg_position"].round(2)
        df.rename(columns={"cluster": "Cluster", "intent": "Intent", "mentions": "Mentions", "avg_position": "Avg Position"}, inplace=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### AI Questions Ranking")
    question_df = fetch_question_ranking(filters)
    if question_df.empty:
        st.info("No AI question data available.")
    else:
        df = question_df.copy()
        df["avg_position"] = df["avg_position"].round(2)
        df.rename(columns={"ai_question": "AI Question", "mentions": "Mentions", "avg_position": "Avg Position"}, inplace=True)
        st.dataframe(df, use_container_width=True, hide_index=True)


def render_ai_sources_tab(filters: FilterState) -> None:
    st.markdown("## AI Sources")
    if not filters.customer_id:
        st.info("Select a customer to explore source data.")
        return

    sources_df = fetch_source_mentions(filters)
    if sources_df.empty:
        st.warning("No source data for the selected filters.")
        return

    df = sources_df.copy()
    df["domain"] = df["url"].apply(extract_domain)
    df = df[df["domain"] != ""]

    if df.empty:
        st.warning("No valid domains found in the selected data.")
        return

    domain_options = sorted(df["domain"].dropna().unique())
    domain_choice = st.selectbox("Select a domain", ["All domains"] + domain_options)
    selected_domain = None if domain_choice == "All domains" else domain_choice

    st.divider()

    if selected_domain:
        domain_df = df[df["domain"] == selected_domain]
        metrics_cols = st.columns(5)
        metrics_cols[0].metric("Your domain mentions", len(domain_df))
        metrics_cols[1].metric("Domain sources", df["domain"].nunique())
        metrics_cols[2].metric("AI Questions", domain_df["ai_question"].nunique())
        metrics_cols[3].metric("Clusters", domain_df["cluster"].nunique())
        metrics_cols[4].metric("LLM", domain_df["llm"].nunique())
    else:
        st.info("Select a domain to unlock detailed metrics.")

    st.markdown("### Domain Sources List")
    domain_list = (
        df.groupby("domain")
        .size()
        .reset_index(name="Mentions")
        .sort_values(by="Mentions", ascending=False)
    )
    st.dataframe(domain_list, use_container_width=True, hide_index=True)

    st.markdown("### URL Sources List")
    url_list = (
        df.groupby("url")
        .size()
        .reset_index(name="Mentions")
        .sort_values(by="Mentions", ascending=False)
    )
    st.dataframe(url_list, use_container_width=True, hide_index=True)

    if selected_domain:
        st.markdown("### Domain Cluster Mentions")
        cluster_table = (
            df[df["domain"] == selected_domain]
            .groupby(["cluster", "intent"])
            .size()
            .reset_index(name="Mentions")
            .sort_values(by="Mentions", ascending=False)
        )
        st.dataframe(cluster_table, use_container_width=True, hide_index=True)

        st.markdown("### Domain Keyword Mentions")
        keyword_table = (
            df[df["domain"] == selected_domain]
            .groupby("keyword")
            .size()
            .reset_index(name="Mentions")
            .sort_values(by="Mentions", ascending=False)
        )
        st.dataframe(keyword_table, use_container_width=True, hide_index=True)

        st.markdown("### AI Questions Mentions")
        question_table = (
            df[df["domain"] == selected_domain]
            .groupby(["ai_question", "tone"])
            .size()
            .reset_index(name="Mentions")
            .sort_values(by="Mentions", ascending=False)
        )
        st.dataframe(question_table, use_container_width=True, hide_index=True)
    else:
        st.info("Detailed tables appear once a domain is selected.")


def main() -> None:
    customer_options = get_customer_options()
    if not customer_options:
        st.warning("No customers found. Please populate the CUSTOMERS mapping.")
        return

    customer_label = st.sidebar.selectbox("Select Customer", list(customer_options.keys()))
    customer_id = customer_options.get(customer_label)

    min_date, max_date = (None, None)
    if customer_id:
        min_date, max_date = fetch_date_bounds(customer_id)

    if min_date and max_date:
        default_range = (min_date, max_date)
        selected_range = st.sidebar.date_input(
            "Date range",
            value=default_range,
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(selected_range, tuple) and len(selected_range) == 2:
            date_range = tuple(sorted(selected_range))
        else:
            date_range = (selected_range, selected_range)
    else:
        date_range = None
        st.sidebar.info("Date filters unavailable for this customer.")

    llm_options = fetch_distinct_column_values(customer_id, "llm") if customer_id else []
    model_options = fetch_distinct_column_values(customer_id, "model") if customer_id else []
    cluster_options = fetch_distinct_column_values(customer_id, "cluster") if customer_id else []
    intent_options = fetch_distinct_column_values(customer_id, "intent") if customer_id else []
    tone_options = fetch_distinct_column_values(customer_id, "tone") if customer_id else []

    selected_llms = st.sidebar.multiselect("LLM", llm_options)
    selected_models = st.sidebar.multiselect("Model", model_options)
    selected_clusters = st.sidebar.multiselect("Cluster", cluster_options)
    selected_intents = st.sidebar.multiselect("Intent", intent_options)
    selected_tones = st.sidebar.multiselect("Tone", tone_options)

    filters = FilterState(
        customer_id=customer_id,
        date_range=date_range,
        llms=tuple(selected_llms),
        models=tuple(selected_models),
        clusters=tuple(selected_clusters),
        intents=tuple(selected_intents),
        tones=tuple(selected_tones),
    )

    brand_tab, sources_tab = st.tabs(["Brand Analysis", "AI Sources"])
    with brand_tab:
        render_brand_analysis_tab(filters)
    with sources_tab:
        render_ai_sources_tab(filters)


if __name__ == "__main__":
    main()
