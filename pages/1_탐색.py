import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="탐색 | 뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/Tl81-go/data/refs/heads/main/stroke.csv"


@st.cache_data(ttl=3600)
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


st.title("🔎 탐색")
st.caption("나이, 혈당, 질환 여부, BMI 결측값, 흡연 상태를 그래프로 살펴봅니다.")

try:
    df = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. GitHub 주소와 CSV 파일을 확인해 주세요.")
    st.code(str(error))
    st.stop()

required_columns = [
    "age",
    "avg_glucose_level",
    "stroke",
    "hypertension",
    "heart_disease",
    "bmi",
    "smoking_status",
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error("CSV 파일에 필요한 열이 없습니다: " + ", ".join(missing_columns))
    st.stop()

data = df.copy()

# 분석에 필요한 열을 숫자형으로 변환
for col in [
    "age",
    "avg_glucose_level",
    "stroke",
    "hypertension",
    "heart_disease",
    "bmi",
]:
    data[col] = pd.to_numeric(data[col], errors="coerce")

data["뇌졸중 여부"] = data["stroke"].map({
    0: "뇌졸중 없음",
    1: "뇌졸중 있음",
})


# ==================================================
# 1. 나이와 평균 혈당의 분포
# ==================================================

st.header("1. 나이와 평균 혈당의 분포")

left, right = st.columns(2)

with left:
    fig_age = px.histogram(
        data.dropna(subset=["age"]),
        x="age",
        nbins=30,
        title="나이 분포",
        labels={
            "age": "나이",
            "count": "사람 수",
        },
    )

    fig_age.update_layout(
        xaxis_title="나이",
        yaxis_title="사람 수",
    )

    st.plotly_chart(fig_age, use_container_width=True)

with right:
    fig_glucose = px.histogram(
        data.dropna(subset=["avg_glucose_level"]),
        x="avg_glucose_level",
        nbins=30,
        title="평균 혈당 분포",
        labels={
            "avg_glucose_level": "평균 혈당",
            "count": "사람 수",
        },
    )

    fig_glucose.update_layout(
        xaxis_title="평균 혈당",
        yaxis_title="사람 수",
    )

    st.plotly_chart(fig_glucose, use_container_width=True)

st.caption(
    "히스토그램은 각 구간에 사람이 얼마나 분포하는지 보여 줍니다."
)

st.divider()


# ==================================================
# 2. 뇌졸중 여부에 따른 나이와 평균 혈당 비교
# ==================================================

st.header("2. 뇌졸중 여부에 따른 나이와 평균 혈당 비교")

stroke_order = ["뇌졸중 없음", "뇌졸중 있음"]

left, right = st.columns(2)

with left:
    age_box = px.box(
        data.dropna(subset=["age", "뇌졸중 여부"]),
        x="뇌졸중 여부",
        y="age",
        category_orders={
            "뇌졸중 여부": stroke_order,
        },
        points="outliers",
        title="뇌졸중 여부별 나이",
        labels={
            "뇌졸중 여부": "그룹",
            "age": "나이",
        },
    )

    st.plotly_chart(age_box, use_container_width=True)

with right:
    glucose_box = px.box(
        data.dropna(
            subset=["avg_glucose_level", "뇌졸중 여부"]
        ),
        x="뇌졸중 여부",
        y="avg_glucose_level",
        category_orders={
            "뇌졸중 여부": stroke_order,
        },
        points="outliers",
        title="뇌졸중 여부별 평균 혈당",
        labels={
            "뇌졸중 여부": "그룹",
            "avg_glucose_level": "평균 혈당",
        },
    )

    st.plotly_chart(glucose_box, use_container_width=True)


# 두 그룹의 평균값 계산
mean_rows = []

for label, value in [
    ("뇌졸중 없음", 0),
    ("뇌졸중 있음", 1),
]:
    group = data[data["stroke"] == value]

    mean_rows.append({
        "그룹": label,
        "평균 나이": group["age"].mean(),
        "평균 혈당": group["avg_glucose_level"].mean(),
    })

mean_table = pd.DataFrame(mean_rows).round(2)

st.subheader("두 그룹의 평균값")

st.dataframe(
    mean_table,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "평균은 해당 값이 비어 있지 않은 사람을 기준으로 계산합니다."
)

st.divider()


# ==================================================
# 3. 고혈압과 심장병에 따른 뇌졸중 비율
# ==================================================

st.header("3. 고혈압과 심장병에 따른 뇌졸중 비율")


def rate_summary(column, labels):
    valid = data.dropna(subset=[column, "stroke"])

    summary = (
        valid.groupby(column)
        .agg(
            사람_수=("stroke", "size"),
            뇌졸중_사람_수=(
                "stroke",
                lambda values: int((values == 1).sum()),
            ),
        )
        .reset_index()
    )

    summary["뇌졸중 비율(%)"] = (
        summary["뇌졸중_사람_수"]
        / summary["사람_수"]
        * 100
    )

    summary["그룹"] = summary[column].map(labels)

    return summary


hypertension = rate_summary(
    "hypertension",
    {
        0: "고혈압 없음",
        1: "고혈압 있음",
    },
)

heart = rate_summary(
    "heart_disease",
    {
        0: "심장병 없음",
        1: "심장병 있음",
    },
)

left, right = st.columns(2)

with left:
    fig = px.bar(
        hypertension,
        x="그룹",
        y="뇌졸중 비율(%)",
        text=hypertension["뇌졸중 비율(%)"].map(
            lambda x: f"{x:.2f}%"
        ),
        title="고혈압 여부별 뇌졸중 비율",
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="뇌졸중 비율(%)",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        hypertension[
            [
                "그룹",
                "사람_수",
                "뇌졸중_사람_수",
                "뇌졸중 비율(%)",
            ]
        ].rename(
            columns={
                "사람_수": "사람 수",
                "뇌졸중_사람_수": "뇌졸중 사람 수",
            }
        ).round(2),
        use_container_width=True,
        hide_index=True,
    )

with right:
    fig = px.bar(
        heart,
        x="그룹",
        y="뇌졸중 비율(%)",
        text=heart["뇌졸중 비율(%)"].map(
            lambda x: f"{x:.2f}%"
        ),
        title="심장병 여부별 뇌졸중 비율",
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="뇌졸중 비율(%)",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        heart[
            [
                "그룹",
                "사람_수",
                "뇌졸중_사람_수",
                "뇌졸중 비율(%)",
            ]
        ].rename(
            columns={
                "사람_수": "사람 수",
                "뇌졸중_사람_수": "뇌졸중 사람 수",
            }
        ).round(2),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "이 비율은 집단별 기록을 비교한 값이며, 인과관계를 뜻하지 않습니다."
)

st.divider()


# ==================================================
# 4. BMI 빈 값인 사람들의 뇌졸중 비율
# ==================================================

st.header("4. BMI 빈 값과 뇌졸중 비율 비교")

missing_bmi = data["bmi"].isna()

missing_group = data.loc[missing_bmi]

valid_all_stroke = data["stroke"].dropna()

valid_missing_stroke = missing_group["stroke"].dropna()


# 전체 뇌졸중 비율
if len(valid_all_stroke) > 0:
    overall_rate = (
        valid_all_stroke.eq(1).mean() * 100
    )
else:
    overall_rate = float("nan")


# BMI가 비어 있는 사람들의 뇌졸중 비율
if len(valid_missing_stroke) > 0:
    missing_rate = (
        valid_missing_stroke.eq(1).mean() * 100
    )
else:
    missing_rate = float("nan")


bmi_table = pd.DataFrame([
    {
        "대상": "BMI가 비어 있는 사람",
        "사람 수": len(missing_group),
        "stroke 확인 가능 인원": len(valid_missing_stroke),
        "뇌졸중 사람 수": int(
            valid_missing_stroke.eq(1).sum()
        ),
        "뇌졸중 비율(%)": missing_rate,
    },
    {
        "대상": "전체 사람",
        "사람 수": len(data),
        "stroke 확인 가능 인원": len(valid_all_stroke),
        "뇌졸중 사람 수": int(
            valid_all_stroke.eq(1).sum()
        ),
        "뇌졸중 비율(%)": overall_rate,
    },
])

bmi_table["뇌졸중 비율(%)"] = (
    bmi_table["뇌졸중 비율(%)"].round(2)
)

st.dataframe(
    bmi_table,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "비율은 stroke 값이 확인되는 사람을 기준으로 계산했습니다."
)

st.divider()


# ==================================================
# 5. 흡연 상태별 사람 수
# ==================================================

st.header("5. 흡연 상태별 사람 수")

smoking_counts = (
    df["smoking_status"]
    .fillna("(빈 값)")
    .astype(str)
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)

st.dataframe(
    smoking_counts,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "이 페이지는 데이터 탐색용이며 개인의 건강 상태를 진단하지 않습니다."
)
