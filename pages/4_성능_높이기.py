
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(
    page_title="성능 높이기 | 뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/Tl81-go/data/refs/heads/main/stroke.csv"

FEATURE_NAMES = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}

DEFAULT_FEATURES = [
    "age",
    "avg_glucose_level",
    "hypertension",
    "heart_disease",
]

MODEL_NAMES = [
    "로지스틱 회귀(확률로 답하는 모델)",
    "의사결정트리(질문으로 답하는 모델)",
]


# --------------------------------------------------
# 데이터 불러오기
# --------------------------------------------------

@st.cache_data(ttl=3600)
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


# --------------------------------------------------
# 평가 지표 계산
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    total = tp + fp + fn + tn

    accuracy = (
        (tp + tn) / total
        if total > 0
        else None
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else None
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else None
    )

    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "정확도": accuracy,
        "재현율": recall,
        "정밀도": precision,
        "F1": f1,
        "전체": total,
    }


def format_metric(value):
    if value is None:
        return "계산할 수 없음"
    return f"{value:.2%}"


# --------------------------------------------------
# 모델별 전후 비교표
# --------------------------------------------------

def make_comparison_table(before, after):
    rows = [
        ("TP · 찾아낸 환자", before["TP"], after["TP"]),
        ("FN · 놓친 환자", before["FN"], after["FN"]),
        ("FP · 헛짚은 사람", before["FP"], after["FP"]),
        ("TN · 정확히 음성으로 판단", before["TN"], after["TN"]),
        (
            "실제 뇌졸중 아님 합계",
            before["TN"] + before["FP"],
            after["TN"] + after["FP"],
        ),
        (
            "실제 뇌졸중 합계",
            before["TP"] + before["FN"],
            after["TP"] + after["FN"],
        ),
        (
            "예측 뇌졸중 아님 합계",
            before["TN"] + before["FN"],
            after["TN"] + after["FN"],
        ),
        (
            "예측 뇌졸중 합계",
            before["TP"] + before["FP"],
            after["TP"] + after["FP"],
        ),
        ("정확도", format_metric(before["정확도"]), format_metric(after["정확도"])),
        ("재현율", format_metric(before["재현율"]), format_metric(after["재현율"])),
        ("정밀도", format_metric(before["정밀도"]), format_metric(after["정밀도"])),
        ("F1", format_metric(before["F1"]), format_metric(after["F1"])),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "항목",
            "가중치 적용 전",
            "가중치 적용 후",
        ],
    )


# --------------------------------------------------
# 모델 생성
# --------------------------------------------------

def make_logistic_model(use_weights):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            random_state=42,
            class_weight="balanced" if use_weights else None,
        ),
    )


def make_tree_model(use_weights):
    return DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
        class_weight="balanced" if use_weights else None,
    )


# --------------------------------------------------
# 2차원 경계선용 데이터 생성
# --------------------------------------------------

def make_2d_grid(
    features,
    x_feature,
    y_feature,
    fixed_values,
    x_grid,
    y_grid,
):
    rows = []

    for y_value in y_grid:
        for x_value in x_grid:
            row = {}

            for feature in features:
                if feature == x_feature:
                    row[feature] = x_value
                elif feature == y_feature:
                    row[feature] = y_value
                else:
                    row[feature] = fixed_values[feature]

            rows.append(row)

    return pd.DataFrame(rows, columns=features)


# --------------------------------------------------
# 3차원 경계면용 데이터 생성
# --------------------------------------------------

def make_3d_grid(
    features,
    x_feature,
    y_feature,
    z_feature,
    fixed_values,
    x_grid,
    y_grid,
    z_grid,
):
    rows = []

    for z_value in z_grid:
        for y_value in y_grid:
            for x_value in x_grid:
                row = {}

                for feature in features:
                    if feature == x_feature:
                        row[feature] = x_value
                    elif feature == y_feature:
                        row[feature] = y_value
                    elif feature == z_feature:
                        row[feature] = z_value
                    else:
                        row[feature] = fixed_values[feature]

                rows.append(row)

    return pd.DataFrame(rows, columns=features)


# --------------------------------------------------
# 화면
# --------------------------------------------------

st.title("🧠 성능 높이기")

st.caption(
    "뇌졸중 환자가 적은 데이터에서 클래스 가중치를 적용하기 전과 "
    "적용한 후의 분류 성능을 비교합니다."
)


# --------------------------------------------------
# 1. 입력 속성 선택
# --------------------------------------------------

st.subheader("1. 입력 속성 고르기")

selected_features = st.multiselect(
    "모델이 사용할 속성",
    options=list(FEATURE_NAMES.keys()),
    default=DEFAULT_FEATURES,
    format_func=lambda key: FEATURE_NAMES[key],
)

if len(selected_features) < 2:
    st.warning("입력 속성을 두 개 이상 골라 주세요.")
    st.stop()


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------

try:
    raw = load_data()

except Exception as error:
    st.error(
        "데이터를 불러오지 못했습니다. "
        "GitHub 주소와 CSV 파일을 확인해 주세요."
    )
    st.code(str(error))
    st.stop()


required_columns = [
    "id",
    "age",
    "avg_glucose_level",
    "bmi",
    "hypertension",
    "heart_disease",
    "stroke",
]

missing_columns = [
    column
    for column in required_columns
    if column not in raw.columns
]

if missing_columns:
    st.error(
        "CSV 파일에 필요한 열이 없습니다: "
        + ", ".join(missing_columns)
    )
    st.stop()


# --------------------------------------------------
# 3. ID 순으로 정렬하고 훈련·테스트 데이터 분리
# --------------------------------------------------

data = raw.copy()

data["id"] = pd.to_numeric(
    data["id"],
    errors="coerce",
)

data["stroke"] = pd.to_numeric(
    data["stroke"],
    errors="coerce",
)

data = data.dropna(
    subset=["id", "stroke"]
)

data = data[
    data["stroke"].isin([0, 1])
]

data = data.sort_values(
    "id",
    kind="stable",
).reset_index(drop=True)

data["stroke"] = data["stroke"].astype(int)

# 10명씩 묶어 앞의 3명은 테스트용으로 분리
data["_position"] = range(len(data))

test_mask = data["_position"] % 10 < 3

train_df = data.loc[~test_mask].copy()
test_df = data.loc[test_mask].copy()

if train_df.empty or test_df.empty:
    st.error("훈련 데이터 또는 테스트 데이터가 비어 있습니다.")
    st.stop()

if train_df["stroke"].nunique() < 2:
    st.error(
        "훈련 데이터에 뇌졸중과 뇌졸중 아님 두 종류가 "
        "모두 있어야 합니다."
    )
    st.stop()


# --------------------------------------------------
# 4. 입력 데이터 숫자 변환 및 결측값 처리
# --------------------------------------------------

X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

for feature in selected_features:
    X_train[feature] = pd.to_numeric(
        X_train[feature],
        errors="coerce",
    )

    X_test[feature] = pd.to_numeric(
        X_test[feature],
        errors="coerce",
    )

    # 훈련 데이터의 중앙값만 사용
    median_value = X_train[feature].median()

    if pd.isna(median_value):
        st.error(
            f"{FEATURE_NAMES[feature]}의 훈련 데이터에 "
            "유효한 숫자가 없습니다."
        )
        st.stop()

    X_train[feature] = X_train[feature].fillna(
        median_value
    )

    X_test[feature] = X_test[feature].fillna(
        median_value
    )

    if feature == "bmi":
        st.write(
            "체질량지수의 빈 값은 훈련 데이터 중앙값 "
            f"**{median_value:.2f}**으로 채웠습니다."
        )

if "bmi" not in selected_features:
    st.caption(
        "체질량지수는 선택하지 않았으므로 BMI 결측값 대체를 하지 않았습니다."
    )

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)

y_train = train_df["stroke"].astype(int).reset_index(drop=True)
y_test = test_df["stroke"].astype(int).reset_index(drop=True)


# --------------------------------------------------
# 5. 데이터 구성 안내
# --------------------------------------------------

positive_train = int((y_train == 1).sum())
negative_train = int((y_train == 0).sum())

positive_test = int((y_test == 1).sum())
negative_test = int((y_test == 0).sum())

st.info(
    f"훈련 데이터: {len(train_df):,}명 "
    f"(뇌졸중 {positive_train:,}명 · "
    f"뇌졸중 아님 {negative_train:,}명)  \n"
    f"테스트 데이터: {len(test_df):,}명 "
    f"(뇌졸중 {positive_test:,}명 · "
    f"뇌졸중 아님 {negative_test:,}명)"
)

if len(test_df) != 1533:
    st.warning(
        f"현재 테스트 데이터는 {len(test_df):,}명입니다. "
        "요청한 1,533명과 다르면 CSV 행 수와 ID 구성을 확인해 주세요."
    )


# --------------------------------------------------
# 6. 가중치 적용 전후 모델 학습
# --------------------------------------------------

st.subheader("2. 가중치 적용 전후 모델 학습")

with st.spinner("네 가지 모델을 학습하고 있습니다..."):

    logistic_before = make_logistic_model(False)
    logistic_after = make_logistic_model(True)

    tree_before = make_tree_model(False)
    tree_after = make_tree_model(True)

    try:
        logistic_before.fit(X_train, y_train)
        logistic_after.fit(X_train, y_train)

        tree_before.fit(X_train, y_train)
        tree_after.fit(X_train, y_train)

    except Exception as error:
        st.error("모델 학습 중 오류가 발생했습니다.")
        st.code(str(error))
        st.stop()


# --------------------------------------------------
# 7. 테스트 데이터 예측
# --------------------------------------------------

logistic_pred_before = logistic_before.predict(X_test)
logistic_pred_after = logistic_after.predict(X_test)

tree_pred_before = tree_before.predict(X_test)
tree_pred_after = tree_after.predict(X_test)


# --------------------------------------------------
# 8. 혼동행렬과 평가 지표 계산
# --------------------------------------------------

logistic_result_before = calculate_metrics(
    y_test,
    logistic_pred_before,
)

logistic_result_after = calculate_metrics(
    y_test,
    logistic_pred_after,
)

tree_result_before = calculate_metrics(
    y_test,
    tree_pred_before,
)

tree_result_after = calculate_metrics(
    y_test,
    tree_pred_after,
)


# --------------------------------------------------
# 9. 전후 비교표
# --------------------------------------------------

st.subheader("3. 가중치 적용 전후 성능 비교")

st.markdown(
    "각 모델에서 가중치를 적용하기 전과 적용한 후의 "
    "혼동행렬 네 칸과 평가 지표를 비교합니다."
)

left, right = st.columns(2)

with left:
    st.markdown(
        "#### 로지스틱 회귀(확률로 답하는 모델)"
    )

    st.dataframe(
        make_comparison_table(
            logistic_result_before,
            logistic_result_after,
        ),
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.markdown(
        "#### 의사결정트리(질문으로 답하는 모델)"
    )

    st.dataframe(
        make_comparison_table(
            tree_result_before,
            tree_result_after,
        ),
        use_container_width=True,
        hide_index=True,
    )


# --------------------------------------------------
# 10. 찾아낸 환자, 놓친 환자, 헛짚은 사람 변화
# --------------------------------------------------

st.subheader("4. 환자 발견 결과 변화")

summary_rows = [
    {
        "모델": "로지스틱 회귀(확률로 답하는 모델)",
        "찾아낸 환자": (
            f"{logistic_result_before['TP']:,}명 → "
            f"{logistic_result_after['TP']:,}명"
        ),
        "놓친 환자": (
            f"{logistic_result_before['FN']:,}명 → "
            f"{logistic_result_after['FN']:,}명"
        ),
        "헛짚은 사람": (
            f"{logistic_result_before['FP']:,}명 → "
            f"{logistic_result_after['FP']:,}명"
        ),
    },
    {
        "모델": "의사결정트리(질문으로 답하는 모델)",
        "찾아낸 환자": (
            f"{tree_result_before['TP']:,}명 → "
            f"{tree_result_after['TP']:,}명"
        ),
        "놓친 환자": (
            f"{tree_result_before['FN']:,}명 → "
            f"{tree_result_after['FN']:,}명"
        ),
        "헛짚은 사람": (
            f"{tree_result_before['FP']:,}명 → "
            f"{tree_result_after['FP']:,}명"
        ),
    },
]

st.dataframe(
    pd.DataFrame(summary_rows),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "찾아낸 환자 = TP · 놓친 환자 = FN · 헛짚은 사람 = FP"
)


# --------------------------------------------------
# 11. 산점도 축 선택
# --------------------------------------------------

st.subheader("5. 가중치 적용 전후 판단 경계 비교")

st.write(
    "점의 색은 실제 결과를 나타냅니다. "
    "두 경계는 같은 테스트 데이터와 같은 다른 속성 값을 기준으로 계산합니다."
)

if len(selected_features) == 2:

    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        format_func=lambda key: FEATURE_NAMES[key],
    )

    y_options = [
        feature
        for feature in selected_features
        if feature != x_feature
    ]

    y_feature = st.selectbox(
        "세로축 속성",
        options=y_options,
        index=0,
        format_func=lambda key: FEATURE_NAMES[key],
    )

    fixed_values = {}

    st.caption(
        "선택한 두 속성을 가로축과 세로축으로 사용합니다."
    )

    # 테스트 데이터 산점도
    plot_df = X_test[[x_feature, y_feature]].copy()

    plot_df["실제 결과"] = y_test.map({
        0: "뇌졸중 아님 (0)",
        1: "뇌졸중 (1)",
    })

    fig = px.scatter(
        plot_df,
        x=x_feature,
        y=y_feature,
        color="실제 결과",
        category_orders={
            "실제 결과": [
                "뇌졸중 아님 (0)",
                "뇌졸중 (1)",
            ],
        },
        labels={
            x_feature: FEATURE_NAMES[x_feature],
            y_feature: FEATURE_NAMES[y_feature],
        },
        title="실제 결과와 가중치 적용 전후 판단 경계",
        opacity=0.65,
    )

    x_min = float(X_test[x_feature].min())
    x_max = float(X_test[x_feature].max())
    y_min = float(X_test[y_feature].min())
    y_max = float(X_test[y_feature].max())

    if x_min == x_max:
        x_min -= 1
        x_max += 1

    if y_min == y_max:
        y_min -= 1
        y_max += 1

    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05

    x_grid = np.linspace(
        x_min - x_padding,
        x_max + x_padding,
        101,
    )

    y_grid = np.linspace(
        y_min - y_padding,
        y_max + y_padding,
        101,
    )

    grid_df = make_2d_grid(
        selected_features,
        x_feature,
        y_feature,
        fixed_values,
        x_grid,
        y_grid,
    )

    # 가중치 적용 전 확률
    logistic_prob_before = logistic_before.predict_proba(
        grid_df
    )[:, 1]

    logistic_prob_before = logistic_prob_before.reshape(
        len(y_grid),
        len(x_grid),
    )

    # 가중치 적용 후 확률
    logistic_prob_after = logistic_after.predict_proba(
        grid_df
    )[:, 1]

    logistic_prob_after = logistic_prob_after.reshape(
        len(y_grid),
        len(x_grid),
    )

    # 의사결정트리의 확률도 동일한 기준으로 계산
    tree_prob_before = tree_before.predict_proba(
        grid_df
    )[:, 1]

    tree_prob_before = tree_prob_before.reshape(
        len(y_grid),
        len(x_grid),
    )

    tree_prob_after = tree_after.predict_proba(
        grid_df
    )[:, 1]

    tree_prob_after = tree_prob_after.reshape(
        len(y_grid),
        len(x_grid),
    )

    # 로지스틱 회귀의 경계선
    fig.add_trace(
        go.Contour(
            x=x_grid,
            y=y_grid,
            z=logistic_prob_before,
            name="로지스틱 회귀 · 가중치 전",
            showscale=False,
            hoverinfo="skip",
            contours={
                "start": 0.5,
                "end": 0.5,
                "size": 0.01,
                "coloring": "none",
                "showlines": True,
            },
            line={
                "color": "blue",
                "width": 3,
                "dash": "solid",
            },
        )
    )

    fig.add_trace(
        go.Contour(
            x=x_grid,
            y=y_grid,
            z=logistic_prob_after,
            name="로지스틱 회귀 · 가중치 후",
            showscale=False,
            hoverinfo="skip",
            contours={
                "start": 0.5,
                "end": 0.5,
                "size": 0.01,
                "coloring": "none",
                "showlines": True,
            },
            line={
                "color": "red",
                "width": 3,
                "dash": "dash",
            },
        )
    )

    # 의사결정트리의 경계선
    fig.add_trace(
        go.Contour(
            x=x_grid,
            y=y_grid,
            z=tree_prob_before,
            name="의사결정트리 · 가중치 전",
            showscale=False,
            hoverinfo="skip",
            contours={
                "start": 0.5,
                "end": 0.5,
                "size": 0.01,
                "coloring": "none",
                "showlines": True,
            },
            line={
                "color": "green",
                "width": 3,
                "dash": "solid",
            },
        )
    )

    fig.add_trace(
        go.Contour(
            x=x_grid,
            y=y_grid,
            z=tree_prob_after,
            name="의사결정트리 · 가중치 후",
            showscale=False,
            hoverinfo="skip",
            contours={
                "start": 0.5,
                "end": 0.5,
                "size": 0.01,
                "coloring": "none",
                "showlines": True,
            },
            line={
                "color": "orange",
                "width": 3,
                "dash": "dash",
            },
        )
    )

    fig.update_layout(
        height=700,
        legend_title_text="모델과 경계선",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # 경계가 화면 범위에 있는지 확인
    boundary_checks = [
        (
            "로지스틱 회귀 · 가중치 전",
            logistic_prob_before,
        ),
        (
            "로지스틱 회귀 · 가중치 후",
            logistic_prob_after,
        ),
        (
            "의사결정트리 · 가중치 전",
            tree_prob_before,
        ),
        (
            "의사결정트리 · 가중치 후",
            tree_prob_after,
        ),
    ]

    for name, probabilities in boundary_checks:
        if not (
            probabilities.min() <= 0.5
            <= probabilities.max()
        ):
            st.caption(
                f"{name}: 0.5 경계선이 현재 그림 범위 밖에 있습니다."
            )


else:

    # --------------------------------------------------
    # 3차원 산점도와 판단 경계면
    # --------------------------------------------------

    st.markdown("#### 3차원 산점도")

    axis_columns = st.columns(3)

    with axis_columns[0]:
        x_feature = st.selectbox(
            "가로축 속성",
            options=selected_features,
            index=0,
            format_func=lambda key: FEATURE_NAMES[key],
        )

    remaining_y = [
        feature
        for feature in selected_features
        if feature != x_feature
    ]

    with axis_columns[1]:
        y_feature = st.selectbox(
            "세로축 속성",
            options=remaining_y,
            index=0,
            format_func=lambda key: FEATURE_NAMES[key],
        )

    remaining_z = [
        feature
        for feature in selected_features
        if feature not in (x_feature, y_feature)
    ]

    with axis_columns[2]:
        z_feature = st.selectbox(
            "깊이축 속성",
            options=remaining_z,
            index=0,
            format_func=lambda key: FEATURE_NAMES[key],
        )

    fixed_values = {
        feature: float(X_test[feature].median())
        for feature in selected_features
        if feature not in (
            x_feature,
            y_feature,
            z_feature,
        )
    }

    if fixed_values:
        fixed_text = " · ".join(
            f"{FEATURE_NAMES[feature]} = {value:.3g}"
            for feature, value in fixed_values.items()
        )

        st.caption(
            "세 축 이외의 속성은 테스트 데이터 중앙값으로 고정했습니다: "
            + fixed_text
        )

    plot_df = X_test[
        [x_feature, y_feature, z_feature]
    ].copy()

    plot_df["실제 결과"] = y_test.map({
        0: "뇌졸중 아님 (0)",
        1: "뇌졸중 (1)",
    })

    fig = px.scatter_3d(
        plot_df,
        x=x_feature,
        y=y_feature,
        z=z_feature,
        color="실제 결과",
        category_orders={
            "실제 결과": [
                "뇌졸중 아님 (0)",
                "뇌졸중 (1)",
            ],
        },
        labels={
            x_feature: FEATURE_NAMES[x_feature],
            y_feature: FEATURE_NAMES[y_feature],
            z_feature: FEATURE_NAMES[z_feature],
        },
        title="실제 결과와 가중치 적용 전후 판단 경계면",
        opacity=0.65,
    )

    # 3차원 격자
    x_min = float(X_test[x_feature].min())
    x_max = float(X_test[x_feature].max())

    y_min = float(X_test[y_feature].min())
    y_max = float(X_test[y_feature].max())

    z_min = float(X_test[z_feature].min())
    z_max = float(X_test[z_feature].max())

    if x_min == x_max:
        x_min -= 1
        x_max += 1

    if y_min == y_max:
        y_min -= 1
        y_max += 1

    if z_min == z_max:
        z_min -= 1
        z_max += 1

    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    z_padding = (z_max - z_min) * 0.05

    # 계산량을 줄이기 위해 3차원 격자는 25칸씩 사용
    x_grid = np.linspace(
        x_min - x_padding,
        x_max + x_padding,
        25,
    )

    y_grid = np.linspace(
        y_min - y_padding,
        y_max + y_padding,
        25,
    )

    z_grid = np.linspace(
        z_min - z_padding,
        z_max + z_padding,
        25,
    )

    grid_df = make_3d_grid(
        selected_features,
        x_feature,
        y_feature,
        z_feature,
        fixed_values,
        x_grid,
        y_grid,
        z_grid,
    )

    # 가중치 적용 전후 확률
    logistic_prob_before = logistic_before.predict_proba(
        grid_df
    )[:, 1]

    logistic_prob_after = logistic_after.predict_proba(
        grid_df
    )[:, 1]

    tree_prob_before = tree_before.predict_proba(
        grid_df
    )[:, 1]

    tree_prob_after = tree_after.predict_proba(
        grid_df
    )[:, 1]

    grid_shape = (
        len(z_grid),
        len(y_grid),
        len(x_grid),
    )

    logistic_prob_before = logistic_prob_before.reshape(
        grid_shape
    )

    logistic_prob_after = logistic_prob_after.reshape(
        grid_shape
    )

    tree_prob_before = tree_prob_before.reshape(
        grid_shape
    )

    tree_prob_after = tree_prob_after.reshape(
        grid_shape
    )

    # 0.5 경계면이 있는지 확인
    boundary_checks = [
        (
            "로지스틱 회귀 · 가중치 전",
            logistic_prob_before,
            "blue",
        ),
        (
            "로지스틱 회귀 · 가중치 후",
            logistic_prob_after,
            "red",
        ),
        (
            "의사결정트리 · 가중치 전",
            tree_prob_before,
            "green",
        ),
        (
            "의사결정트리 · 가중치 후",
            tree_prob_after,
            "orange",
        ),
    ]

    for name, probabilities, color in boundary_checks:

        if (
            probabilities.min() <= 0.5
            <= probabilities.max()
        ):
            fig.add_trace(
                go.Isosurface(
                    x=np.tile(
                        x_grid,
                        len(y_grid) * len(z_grid),
                    ),
                    y=np.tile(
                        np.repeat(y_grid, len(x_grid)),
                        len(z_grid),
                    ),
                    z=np.repeat(
                        z_grid,
                        len(y_grid) * len(x_grid),
                    ),
                    value=probabilities.flatten(),
                    isomin=0.5,
                    isomax=0.5,
                    surface_count=1,
                    opacity=0.22,
                    colorscale=[
                        [0, color],
                        [1, color],
                    ],
                    showscale=False,
                    caps={
                        "x_show": False,
                        "y_show": False,
                        "z_show": False,
                    },
                    name=name,
                    hoverinfo="skip",
                )
            )

        else:
            st.caption(
                f"{name}: 0.5 경계면이 현재 그림 범위 밖에 있습니다."
            )

    fig.update_layout(
        height=800,
        scene={
            "xaxis_title": FEATURE_NAMES[x_feature],
            "yaxis_title": FEATURE_NAMES[y_feature],
            "zaxis_title": FEATURE_NAMES[z_feature],
        },
        legend_title_text="실제 결과와 경계면",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "마우스로 드래그하면 3차원 그림을 회전할 수 있습니다. "
        "반투명한 색 면은 각 모델의 확률 0.5 경계면입니다."
    )


# --------------------------------------------------
# 마무리 안내
# --------------------------------------------------

st.divider()

st.markdown("### 결과를 해석할 때 주의할 점")

st.write(
    "클래스 가중치를 적용하면 드문 뇌졸중 양성 사례를 "
    "모델이 더 중요하게 학습하도록 할 수 있습니다. "
    "그 결과 재현율이 높아질 수 있지만, 헛짚은 사람(FP)이 "
    "늘어나 정밀도나 정확도가 낮아질 수도 있습니다."
)

st.write(
    "따라서 정확도 하나만 보지 말고, "
    "놓친 환자(FN), 헛짚은 사람(FP), 재현율, 정밀도, F1을 "
    "함께 비교해야 합니다."
)

st.caption(
    "이 페이지는 분류 학습을 위한 실습입니다. "
    "실제 의료 진단이나 치료 판단에 사용할 수 있는 모델은 아닙니다."
)
