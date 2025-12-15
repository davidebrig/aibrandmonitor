from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse

import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine

# Initialize DB engine
# We use st.secrets so this assumes secrets are set up
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

    df = fetch_customers()
    if df.empty:
        return {}

    # key = readable name, value = id (UUID)
    return {row.name: str(row.id) for row in df.itertuples()}


@st.cache_data(ttl=300)
def fetch_customers() -> pd.DataFrame:
    sql = """
        SELECT id, name
        FROM customers
        WHERE id IS NOT NULL
        ORDER BY name
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


def fetch_ai_questions_analytics(filters: FilterState, my_brand: str, my_domain: str) -> pd.DataFrame:
    """
    Aggregates metrics per AI Question.
    """
    where_clause, params = build_where_clause(filters)
    
    # 1. Brand Metrics from v_brand_mentions_flat
    brand_sql = f"""
        SELECT 
            ai_question,
            SUM(CASE WHEN brand = %(my_brand)s THEN 1 ELSE 0 END) as my_brand_mentions,
            COUNT(DISTINCT brand) as total_brand_mentions
        FROM v_brand_mentions_flat
        WHERE {where_clause}
        GROUP BY ai_question
    """
    brand_params = params.copy()
    brand_params["my_brand"] = my_brand
    df_brands = run_query(brand_sql, brand_params)
    
    # 2. Source Metrics from v_source_mentions_flat
    sources_sql = f"""
        SELECT 
            ai_question,
            url,
            keyword,
            volume
        FROM v_source_mentions_flat
        WHERE {where_clause}
          AND url IS NOT NULL
    """
    df_sources = run_query(sources_sql, params)
    
    if not df_sources.empty:
        df_sources["domain"] = df_sources["url"].apply(extract_domain)
        
        # Intent Volume
        df_unique_kw = df_sources[["ai_question", "keyword", "volume"]].drop_duplicates()
        vol_per_q = df_unique_kw.groupby("ai_question")["volume"].sum().reset_index(name="intent_volume")
        
        # Citations
        df_sources["is_my_domain"] = df_sources["domain"] == my_domain
        citations_per_q = df_sources.groupby("ai_question").agg(
            my_domain_citations=("is_my_domain", "sum"),
            total_citations=("url", "count")
        ).reset_index()
        
        df_b = pd.merge(vol_per_q, citations_per_q, on="ai_question", how="outer")
    else:
        df_b = pd.DataFrame(columns=["ai_question", "intent_volume", "my_domain_citations", "total_citations"])

    # Merge
    final_df = pd.merge(df_brands, df_b, on="ai_question", how="outer").fillna(0)
    
    # Relevance
    final_df["relevance_pct"] = final_df.apply(
        lambda x: (x["my_brand_mentions"] / x["total_brand_mentions"]) * 100 if x["total_brand_mentions"] > 0 else 0,
        axis=1
    )
    
    return final_df


def fetch_latest_ai_responses(filters: FilterState, question: str) -> pd.DataFrame:
    """Gets the latest raw response for each LLM for a specific question."""
    where_clause, params = build_where_clause(filters)
    params = params.copy()
    params["question"] = question
    
    # We want the latest response text per LLM
    sql = f"""
        SELECT DISTINCT ON (llm) 
            llm, 
            date, 
            risposta as response
        FROM v_ai_responses_flat
        WHERE {where_clause}
          AND ai_question = %(question)s
          AND risposta IS NOT NULL
        ORDER BY llm, date DESC
    """
    return run_query(sql, params)


def fetch_question_sources(filters: FilterState, question: str) -> pd.DataFrame:
    """Gets list of sources for a specific question."""
    where_clause, params = build_where_clause(filters)
    params = params.copy()
    params["question"] = question
    
    sql = f"""
        SELECT date, url, llm
        FROM v_source_mentions_flat
        WHERE {where_clause}
          AND ai_question = %(question)s
          AND url IS NOT NULL
        ORDER BY date DESC
    """
    return run_query(sql, params)


def fetch_question_brand_timeline(filters: FilterState, question: str, brand: str) -> pd.DataFrame:
    """Gets position timeline for brand on a specific question."""
    where_clause, params = build_where_clause(filters)
    params = params.copy()
    params["question"] = question
    params["brand"] = brand
    
    sql = f"""
        SELECT date, llm, position
        FROM v_brand_mentions_flat
        WHERE {where_clause}
          AND ai_question = %(question)s
          AND brand = %(brand)s
          AND position IS NOT NULL
        ORDER BY date
    """
    return run_query(sql, params)


def login_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Logs in a user via Supabase Auth (REST API).
    Returns the user object (including access_token) if successful, else None.
    """
    # Assuming secrets are available as st.secrets["supabase"]["project_url"] and generic anon key
    # If using different structure, adjust accordingly.
    try:
        project_url = st.secrets["supabase"]["project_url"]
        api_key = st.secrets["supabase"]["anon_key"]
    except KeyError:
        st.error("Supabase secrets not found. Please check .streamlit/secrets.toml")
        return None
    
    auth_url = f"{project_url}/auth/v1/token?grant_type=password"
    headers = {
        "apikey": api_key,
        "Content-Type": "application/json",
    }
    data = {
        "email": email,
        "password": password,
    }
    
    try:
        response = requests.post(auth_url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            st.warning(f"Login failed: {response.json().get('error_description', 'Invalid credentials')}")
            return None
    except Exception as e:
        st.error(f"An error occurred during login: {e}")
        return None


def get_user_customer_id(user_id: str) -> Optional[str]:
    """
    Fetches the customer_id associated with the user.
    """
    sql = """
        SELECT customer_id
        FROM user_customers
        WHERE user_id = %(user_id)s
    """
    try:
        df = run_query(sql, {"user_id": user_id})
        if not df.empty and pd.notnull(df.iloc[0]["customer_id"]):
            return str(df.iloc[0]["customer_id"])
    except Exception:
        pass
    return None
