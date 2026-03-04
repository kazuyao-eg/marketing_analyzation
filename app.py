import pandas as pd
import numpy as np
import streamlit as st
import altair as alt

st.set_page_config(
    page_title="流入分析",
    layout="wide"
)

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["FC実施年月日"] = pd.to_datetime(df["FC実施年月日"], errors="coerce")
    df["年月"] = df["FC実施年月日"].dt.to_period("M").astype(str)

    df["入会フラグ"] = np.where(df["ステータス"] == "入会", 1, 0)
    df["入会ステータス"] = np.where(df["入会フラグ"] == 1, "入会", "非入会")

    age_order = ["10代前半", "18〜25", "26〜30", "31〜35", "36〜45", "46〜60", "61以上", "不明"]
    df["年代"] = pd.Categorical(df["年代"], categories=age_order, ordered=True)

    return df


def apply_filters(df: pd.DataFrame) -> tuple:
    st.sidebar.header("フィルタ")

    month_list = (
        df["年月"]
        .dropna()
        .sort_values()
        .unique()
        .tolist()
    )
    if len(month_list) == 0:
        st.error("年月データが存在しません。`FC実施年月日` の形式を確認してください。")
        return df, None

    start_month, end_month = st.sidebar.select_slider(
        "表示期間（年月）",
        options=month_list,
        value=(month_list[0], month_list[-1])
    )

    month_mask = (df["年月"] >= start_month) & (df["年月"] <= end_month)
    df = df.loc[month_mask].copy()

    genders = df["性別"].dropna().unique().tolist()
    selected_genders = st.sidebar.multiselect(
        "性別",
        options=genders,
        default=genders
    )
    if selected_genders:
        df = df[df["性別"].isin(selected_genders)]

    ages = df["年代"].dropna().unique().tolist()
    selected_ages = st.sidebar.multiselect(
        "年代",
        options=ages,
        default=ages
    )
    if selected_ages:
        df = df[df["年代"].isin(selected_ages)]

    countries = df["在住国"].dropna().unique().tolist()
    selected_countries = st.sidebar.multiselect(
        "在住国",
        options=countries,
        default=countries
    )
    if selected_countries:
        df = df[df["在住国"].isin(selected_countries)]

    cefrs = df["CEFR"].dropna().unique().tolist()
    selected_cefrs = st.sidebar.multiselect(
        "CEFR",
        options=cefrs,
        default=cefrs
    )
    if selected_cefrs:
        df = df[df["CEFR"].isin(selected_cefrs)]

    channel_axis = st.sidebar.radio(
        "チャネル軸の選択",
        options=["集客経路", "流入経路", "識別用のラベル"],
        index=0,
        help="標準は『集客経路』。必要に応じて他の軸でも分析できます。"
    )

    top_n = st.sidebar.slider(
        "チャネル表示数（上位 N 件）",
        min_value=5,
        max_value=30,
        value=10,
        step=1
    )

    min_count = st.sidebar.slider(
        "チャネルの最低 FC 件数（この件数未満は『その他』に集約）",
        min_value=1,
        max_value=50,
        value=5,
        step=1
    )

    filters = {
        "start_month": start_month,
        "end_month": end_month,
        "channel_axis": channel_axis,
        "top_n": top_n,
        "min_count": min_count,
    }

    return df, filters


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("年月")
        .agg(
            FC件数=("ステータス", "size"),
            入会件数=("入会フラグ", "sum"),
        )
        .reset_index()
    )
    agg["入会率"] = np.where(
        agg["FC件数"] > 0,
        agg["入会件数"] / agg["FC件数"],
        np.nan
    )
    return agg


def aggregate_segment(df: pd.DataFrame, col: str) -> pd.DataFrame:
    base = (
        df.groupby([col, "入会ステータス"])
        .agg(件数=("ステータス", "size"))
        .reset_index()
    )

    total = (
        base.groupby(col)["件数"]
        .sum()
        .reset_index()
        .rename(columns={"件数": "総数"})
    )

    merged = base.merge(total, on=col, how="left")
    merged["比率"] = merged["件数"] / merged["総数"]
    return merged


def aggregate_channel(df: pd.DataFrame, col: str, top_n: int, min_count: int) -> pd.DataFrame:
    base = (
        df.groupby([col])
        .agg(
            FC件数=("ステータス", "size"),
            入会件数=("入会フラグ", "sum"),
        )
        .reset_index()
    )
    base["入会率"] = np.where(
        base["FC件数"] > 0,
        base["入会件数"] / base["FC件数"],
        np.nan
    )

    major = base[base["FC件数"] >= min_count].copy()
    minor = base[base["FC件数"] < min_count].copy()

    if not minor.empty:
        other_row = pd.DataFrame([{
            col: "その他（少数チャネル）",
            "FC件数": minor["FC件数"].sum(),
            "入会件数": minor["入会件数"].sum()
        }])
        other_row["入会率"] = np.where(
            other_row["FC件数"] > 0,
            other_row["入会件数"] / other_row["FC件数"],
            np.nan
        )
        base = pd.concat([major, other_row], ignore_index=True)
    else:
        base = major

    base = base.sort_values("FC件数", ascending=False)
    top = base.head(top_n).copy()
    rest = base.iloc[top_n:].copy()

    if not rest.empty:
        rest_row = pd.DataFrame([{
            col: "その他（上位外）",
            "FC件数": rest["FC件数"].sum(),
            "入会件数": rest["入会件数"].sum()
        }])
        rest_row["入会率"] = np.where(
            rest_row["FC件数"] > 0,
            rest_row["入会件数"] / rest_row["FC件数"],
            np.nan
        )
        top = pd.concat([top, rest_row], ignore_index=True)

    return top


def aggregate_cefr(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby("CEFR")
        .agg(
            FC件数=("ステータス", "size"),
            入会件数=("入会フラグ", "sum"),
        )
        .reset_index()
    )
    agg["入会率"] = np.where(
        agg["FC件数"] > 0,
        agg["入会件数"] / agg["FC件数"],
        np.nan
    )
    return agg


def main():
    st.title("スポーツジム 無料カウンセリング / 入会ダッシュボード")

    st.markdown(
        "入会 vs 非入会 にフォーカスして、"
        "月別の推移・属性別の構成・チャネル別・CEFR別の入会率を把握するためのダッシュボードです。"
    )

    try:
        df_raw = load_data("fc_info.csv")
    except FileNotFoundError:
        st.error("`fc_info.csv` が見つかりません。`app.py` と同じフォルダに配置してください。")
        return

    df_filtered, filters = apply_filters(df_raw)

    if filters is None:
        return

    if df_filtered.empty:
        st.warning("現在のフィルタ条件ではデータがありません。条件を緩めてみてください。")
        return

    tab_summary, tab_segment, tab_cefr = st.tabs(["サマリー", "流入像（属性・チャネル）", "CEFR分析"])

    # ===== サマリータブ =====
    with tab_summary:
        st.subheader("KPIサマリー")

        total_fc = len(df_filtered)
        total_member = int(df_filtered["入会フラグ"].sum())
        conv_rate = total_member / total_fc if total_fc > 0 else np.nan

        col1, col2, col3 = st.columns(3)
        col1.metric("FC件数（レコード数）", f"{total_fc:,}")
        col2.metric("入会件数", f"{total_member:,}")
        col3.metric(
            "入会率",
            f"{conv_rate * 100:,.1f}%" if not np.isnan(conv_rate) else "―"
        )

        st.markdown("---")
        st.subheader("月別 FC件数 / 入会件数 / 入会率")

        monthly = aggregate_monthly(df_filtered)

        base = alt.Chart(monthly).encode(
            x=alt.X("年月:N", sort=monthly["年月"].tolist(), title="年月")
        )

        bar_fc = base.mark_bar(color="#4C78A8", opacity=0.8).encode(
            y=alt.Y("FC件数:Q", axis=alt.Axis(title="FC件数"))
        )

        line_conv = base.mark_line(color="#F58518", point=True).encode(
            y=alt.Y("入会率:Q", axis=alt.Axis(title="入会率", format=".0%")),
        )

        chart = alt.layer(
            bar_fc,
            line_conv.encode(y="入会率:Q")
        ).resolve_scale(
            y="independent"
        )

        st.altair_chart(chart, use_container_width=True)

        st.markdown("---")
        st.subheader("属性構成（参考）")

        col_a, col_b = st.columns(2)

        with col_a:
            st.caption("性別構成")
            gender_dist = (
                df_filtered["性別"]
                .value_counts(dropna=False)
                .reset_index()
            )
            gender_dist.columns = ["性別", "件数"]
            gender_dist["比率"] = gender_dist["件数"] / gender_dist["件数"].sum()
            st.dataframe(gender_dist)

        with col_b:
            st.caption("在住国構成")
            country_dist = (
                df_filtered["在住国"]
                .value_counts(dropna=False)
                .reset_index()
            )
            country_dist.columns = ["在住国", "件数"]
            country_dist["比率"] = country_dist["件数"] / country_dist["件数"].sum()
            st.dataframe(country_dist)

    # ===== 流入像（属性・チャネル）タブ =====
    with tab_segment:
        st.subheader("属性別：入会 vs 非入会")

        display_mode = st.radio(
            "表示形式の切り替え",
            options=["絶対数（件数）", "割合（構成比）"],
            horizontal=True
        )

        seg_gender = aggregate_segment(df_filtered, "性別")
        seg_gender_chart = alt.Chart(seg_gender).mark_bar().encode(
            x=alt.X("性別:N", title="性別"),
            y=alt.Y(
                "件数:Q" if display_mode == "絶対数（件数）" else "比率:Q",
                axis=alt.Axis(title="件数" if display_mode == "絶対数（件数）" else "構成比", format=",.0f" if display_mode == "絶対数（件数）" else ".0%")
            ),
            color=alt.Color("入会ステータス:N", title="ステータス"),
            tooltip=["性別", "入会ステータス", "件数", alt.Tooltip("比率", format=".1%")]
        ).properties(
            title="性別別 入会 vs 非入会"
        )
        st.altair_chart(seg_gender_chart, use_container_width=True)

        seg_age = aggregate_segment(df_filtered, "年代")
        seg_age_chart = alt.Chart(seg_age).mark_bar().encode(
            x=alt.X("年代:N", title="年代", sort=seg_age["年代"].cat.categories.tolist() if hasattr(seg_age["年代"], "cat") else alt.SortField("年代")),
            y=alt.Y(
                "件数:Q" if display_mode == "絶対数（件数）" else "比率:Q",
                axis=alt.Axis(title="件数" if display_mode == "絶対数（件数）" else "構成比", format=",.0f" if display_mode == "絶対数（件数）" else ".0%")
            ),
            color=alt.Color("入会ステータス:N", title="ステータス"),
            tooltip=["年代", "入会ステータス", "件数", alt.Tooltip("比率", format=".1%")]
        ).properties(
            title="年代別 入会 vs 非入会"
        )
        st.altair_chart(seg_age_chart, use_container_width=True)

        st.markdown("---")
        st.subheader(f"チャネル別（{filters['channel_axis']}） 入会分析")

        channel_col = filters["channel_axis"]
        channel_agg = aggregate_channel(
            df_filtered,
            col=channel_col,
            top_n=filters["top_n"],
            min_count=filters["min_count"]
        )

        st.dataframe(channel_agg.sort_values("FC件数", ascending=False))

        channel_chart = alt.Chart(channel_agg).encode(
            x=alt.X(f"{channel_col}:N", sort="-y", title=channel_col),
        )

        bar_fc = channel_chart.mark_bar(color="#4C78A8").encode(
            y=alt.Y("FC件数:Q", axis=alt.Axis(title="FC件数"))
        )

        line_conv = channel_chart.mark_line(color="#F58518", point=True).encode(
            y=alt.Y("入会率:Q", axis=alt.Axis(title="入会率", format=".0%")),
        )

        layered = alt.layer(bar_fc, line_conv).resolve_scale(
            y="independent"
        ).properties(
            title=f"{channel_col} 別 FC件数 / 入会率（上位 {filters['top_n']} ＋ その他）"
        )

        st.altair_chart(layered, use_container_width=True)

    # ===== CEFR分析タブ =====
    with tab_cefr:
        st.subheader("CEFR別 入会分析")

        cefr_agg = aggregate_cefr(df_filtered)
        st.dataframe(cefr_agg.sort_values("FC件数", ascending=False))

        cefr_chart = alt.Chart(cefr_agg).encode(
            x=alt.X("CEFR:N", title="CEFR")
        )

        bar_fc = cefr_chart.mark_bar(color="#4C78A8").encode(
            y=alt.Y("FC件数:Q", axis=alt.Axis(title="FC件数"))
        )

        line_conv = cefr_chart.mark_line(color="#F58518", point=True).encode(
            y=alt.Y("入会率:Q", axis=alt.Axis(title="入会率", format=".0%"))
        )

        cefr_layered = alt.layer(bar_fc, line_conv).resolve_scale(
            y="independent"
        ).properties(
            title="CEFR別 FC件数 / 入会率"
        )

        st.altair_chart(cefr_layered, use_container_width=True)


if __name__ == "__main__":
    main()
