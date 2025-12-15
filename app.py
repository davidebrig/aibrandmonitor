from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

# Import shared utilities
from utils import (
    FilterState,
    build_where_clause,
    extract_domain,
    fetch_ai_questions_analytics,
    fetch_brand_list,
    fetch_customers,
    fetch_date_bounds,
    fetch_distinct_column_values,
    fetch_latest_ai_responses,
    fetch_question_brand_timeline,
    fetch_question_sources,
    format_avg,
    get_customer_options,
    get_user_customer_id,
    login_user,
    run_query,
    safe_int,
)

st.set_page_config(page_title="AI Brand Monitor Dashboard", layout="wide")
st.title("AI Brand Monitor Dashboard")




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


def render_ai_questions_tab(filters: FilterState) -> None:
    st.markdown("## AI Questions Analysis")
    if not filters.customer_id:
        st.info("Select a customer to explore AI Questions.")
        return

    st.markdown("### Configuration")
    col1, col2 = st.columns(2)
    
    brand_options = fetch_brand_list(filters)
    default_brand_index = 0
    selected_brand = col1.selectbox("Select My Brand", brand_options, index=default_brand_index)
    
    # Fetch domains for selection (using utils logic explicitly or run query)
    where_clause, params = build_where_clause(filters)
    source_df = run_query(f"SELECT url FROM v_source_mentions_flat WHERE {where_clause} AND url IS NOT NULL", params)
    
    if not source_df.empty:
        # We need extract_domain again? It is in utils, so we import it if needed or just use logic.
        # But wait, I didn't import extract_domain in app.py in the previous step... 
        # Actually I did import it in the Refactor step but removed it from app.py definition.
        # Let's check imports. `from utils import ... extract_domain ...`
        source_df["domain"] = source_df["url"].apply(extract_domain)
        all_domains = sorted(source_df["domain"].unique().tolist())
    else:
        all_domains = []
        
    selected_domain = col2.selectbox("Select My Domain", all_domains)

    if not selected_brand or not selected_domain:
        st.info("Select Brand and Domain to see analytics.")
        return
        
    st.divider()
    
    with st.spinner("Analyzing AI Questions..."):
        df_analytics = fetch_ai_questions_analytics(filters, selected_brand, selected_domain)
    
    if df_analytics.empty:
        st.warning("No data found for the selected filters.")
        return

    # Rename columns for display
    display_df = df_analytics.rename(columns={
        "ai_question": "AI Question",
        "intent_volume": "Intent Volume",
        "my_brand_mentions": "My Brand Mentions",
        "total_brand_mentions": "Total Brand Mentions",
        "relevance_pct": "% Relevance",
        "my_domain_citations": "My Domain Citations",
        "total_citations": "Total Citations"
    })
    
    display_df["Intent Volume"] = display_df["Intent Volume"].astype(int)
    display_df["My Brand Mentions"] = display_df["My Brand Mentions"].astype(int)
    display_df["Total Brand Mentions"] = display_df["Total Brand Mentions"].astype(int)
    display_df["My Domain Citations"] = display_df["My Domain Citations"].astype(int)
    display_df["Total Citations"] = display_df["Total Citations"].astype(int)
    display_df["% Relevance"] = display_df["% Relevance"].map("{:.2f}%".format)

    cols_order = [
        "AI Question", 
        "Intent Volume", 
        "My Brand Mentions", 
        "Total Brand Mentions", 
        "% Relevance", 
        "My Domain Citations", 
        "Total Citations"
    ]
    display_df = display_df[cols_order]

    # --- Interactive Details ---
    # Need to access selection state. Streamlit dataframe with on_select returns a selection object.
    # To make it work nicely, we should use session state or just check the event.
    
    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
        key="ai_questions_table",
        column_config={
            "Intent Volume": st.column_config.NumberColumn(help="Sum of search volumes of associated keywords"),
            "Total Brand Mentions": st.column_config.NumberColumn(help="Distinct count of brands mentioned"),
            "% Relevance": st.column_config.TextColumn(help="My Brand Mentions / Total Brand Mentions (Distinct)"),
        }
    )

    if event.selection and event.selection.rows:
        selected_index = event.selection.rows[0]
        selected_row = display_df.iloc[selected_index]
        selected_question = selected_row["AI Question"]
        
        st.divider()
        st.markdown(f"### Details for: *{selected_question}*")
        
        # 1. Timeline
        st.markdown("#### Brand Timeline")
        timeline_df = fetch_question_brand_timeline(filters, selected_question, selected_brand)
        if not timeline_df.empty:
            chart = (
                alt.Chart(timeline_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("position:Q", title="Position", scale=alt.Scale(reverse=True)), # Rank 1 is top
                    color=alt.Color("llm:N", title="LLM"),
                    tooltip=["date", "llm", "position"]
                )
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No timeline data for this brand/question.")

        # 2. Sources
        st.markdown("#### Cited Sources")
        sources_df = fetch_question_sources(filters, selected_question)
        if not sources_df.empty:
            sources_df["domain"] = sources_df["url"].apply(extract_domain)
            st.dataframe(sources_df[["date", "llm", "domain", "url"]], use_container_width=True, hide_index=True)
        else:
            st.info("No sources found.")

        # 3. Raw Responses
        st.markdown("#### Latest Raw Responses")
        responses_df = fetch_latest_ai_responses(filters, selected_question)
        if not responses_df.empty:
            for row in responses_df.itertuples():
                with st.expander(f"{row.llm} ({row.date})"):
                    st.markdown(row.response)
        else:
            st.info("No raw responses available.")


def render_login() -> None:
    st.markdown("## Login to AI Brand Monitor")
    
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if not email or not password:
                st.error("Please enter both email and password.")
                return
            
            with st.spinner("Authenticating..."):
                user = login_user(email, password)
                if user:
                    # Successful login
                    user_id = user["user"]["id"]
                    access_token = user["access_token"]
                    
                    # Resolve Customer
                    customer_id = get_user_customer_id(user_id)
                    
                    if not customer_id:
                        st.error("Login successful, but no customer is associated with this account. Please contact support.")
                        return

                    # Save to Session State
                    st.session_state["user_id"] = user_id
                    st.session_state["user_email"] = user["user"]["email"]
                    st.session_state["access_token"] = access_token
                    st.session_state["customer_id"] = customer_id
                    st.session_state["logged_in"] = True
                    st.success("Login successful! Redirecting...")
                    st.rerun()


def main() -> None:
    # 1. Check Authentication
    if not st.session_state.get("logged_in"):
        render_login()
        return

    # 2. Authenticated Flow
    # Logout Button in Sidebar
    with st.sidebar:
        st.image("assets/logo.png", use_container_width=True)
        st.divider()
        st.write(f"Ciao **{st.session_state.get('user_email', 'User')}**")
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()
            
    customer_id = st.session_state.get("customer_id")
    if not customer_id:
        st.error("Session invalid. Please login again.")
        st.session_state.clear()
        return

    # 3. Dashboard Logic (No Customer Selector)
    
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

    # Note: customer_id is fixed from session
    filters = FilterState(
        customer_id=customer_id,
        date_range=date_range,
        llms=tuple(selected_llms),
        models=tuple(selected_models),
        clusters=tuple(selected_clusters),
        intents=tuple(selected_intents),
        tones=tuple(selected_tones),
    )

    brand_tab, sources_tab, questions_tab = st.tabs(["Brand Analysis", "AI Sources", "AI Questions"])
    with brand_tab:
        render_brand_analysis_tab(filters)
    with sources_tab:
        render_ai_sources_tab(filters)
    with questions_tab:
        render_ai_questions_tab(filters)


if __name__ == "__main__":
    main()
