import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


st.set_page_config(
    page_title="채점표 | 뇌졸중 예측 실습실",
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


@st.cache_data(ttl=3600)
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def calculate_metrics(y_true, y_pred):
    """
    혼동행렬의 네 값과 네 가지 평가 지표를 계산한다.

    양성: 뇌졸중(1)
    음성: 뇌졸중 아님(0)
    """

    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

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
        "total": total,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def format_metric(value):
    if value is None:
        return "계산할 수 없음"
    return f"{value:.2%}"


def show_confusion_matrix(title, result):
    """
    혼동행렬을 2×2 표와 합계로 표시한다.
    행은 실제, 열은 예측이다.
    """

    st.markdown(f"#### {title}")

    matrix_df = pd.DataFrame(
        [
            [
                f"TN · {result['TN']:,}명",
                f"FP · {result['FP']:,}명",
                f"{result['TN'] + result['FP']:,}명",
            ],
            [
                f"FN · {result['FN']:,}명",
                f"TP · {result['TP']:,}명",
                f"{result['FN'] + result['TP']:,}명",
            ],
            [
                f"{result['TN'] + result['FN']:,}명",
                f"{result['FP'] + result['TP']:,}명",
                f"{result['total']:,}명",
            ],
        ],
        index=[
            "실제 뇌졸중 아님 (0)",
            "실제 뇌졸중 (1)",
            "합계",
        ],
        columns=[
            "예측 뇌졸중 아님 (0)",
            "예측 뇌졸중 (1)",
            "합계",
        ],
    )

    st.table(matrix_df)

    st.caption(
        "TN: 실제 음성·예측 음성 / "
        "FP: 실제 음성·예측 양성 / "
        "FN: 실제 양성·예측 음성 / "
        "TP: 실제 양성·예측 양성"
    )


def make_metric_row(name, result):
    return {
        "모델": name,
        "정확도": format_metric(result["accuracy"]),
        "재현율": format_metric(result["recall"]),
        "정밀도": format_metric(result["precision"]),
        "F1": format_metric(result["f1"]),
    }


def make_person_list(test_df, y_true, y_pred, error_type):
    """
    FN 또는 FP에 해당하는 사람들의 목록을 만든다.

    FN: 실제 1, 예측 0
    FP: 실제 0, 예측 1
    """

    people = test_df.copy()

    people["_actual"] = list(y_true)
    people["_predicted"] = list(y_pred)

    if error_type == "FN":
        people = people[
            (people["_actual"] == 1)
            & (people["_predicted"] == 0)
        ].copy()

    elif error_type == "FP":
        people = people[
            (people["_actual"] == 0)
            & (people["_predicted"] == 1)
        ].copy()

    people["id"] = pd.to_numeric(
        people["id"],
        errors="coerce",
    )

    people["age"] = pd.to_numeric(
        people["age"],
        errors="coerce",
    )

    people["avg_glucose_level"] = pd.to_numeric(
        people["avg_glucose_level"],
        errors="coerce",
    )

    people["hypertension"] = pd.to_numeric(
        people["hypertension"],
        errors="coerce",
    )

    people["heart_disease"] = pd.to_numeric(
        people["heart_disease"],
        errors="coerce",
    )

    people = people.sort_values(
        "age",
        ascending=False,
        na_position="last",
    )

    display_df = people[
        [
            "id",
            "age",
            "avg_glucose_level",
            "hypertension",
            "heart_disease",
        ]
    ].copy()

    display_df = display_df.rename(
        columns={
            "id": "번호",
            "age": "나이",
            "avg_glucose_level": "평균 혈당",
            "hypertension": "고혈압 여부",
            "heart_disease": "심장병 여부",
        }
    )

    display_df["고혈압 여부"] = display_df[
        "고혈압 여부"
    ].map({
        0: "아니요",
        1: "예",
    })

    display_df["심장병 여부"] = display_df[
        "심장병 여부"
    ].map({
        0: "아니요",
        1: "예",
    })

    return display_df, len(people)


# --------------------------------------------------
# 화면 시작
# --------------------------------------------------

st.title("🧠 채점표")

st.caption(
    "테스트 데이터의 실제 정답과 모델의 예측을 비교합니다. "
    "뇌졸중(1)을 양성, 뇌졸중 아님(0)을 음성으로 정합니다."
)


# --------------------------------------------------
# 1. 데이터 불러오기
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
# 2. 데이터 정렬 및 훈련·테스트 분리
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

# 10명씩 묶어 앞의 3명은 테스트용, 나머지 7명은 훈련용
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
        "모두 있어야 모델을 학습할 수 있습니다."
    )
    st.stop()


# --------------------------------------------------
# 3. 입력 속성 선택
# --------------------------------------------------

st.subheader("1. 모델 입력 속성")

selected_features = st.multiselect(
    "분류 모델 페이지와 동일하게 사용할 속성을 선택하세요.",
    options=list(FEATURE_NAMES.keys()),
    default=DEFAULT_FEATURES,
    format_func=lambda key: FEATURE_NAMES[key],
)

if len(selected_features) < 2:
    st.warning("입력 속성을 두 개 이상 골라 주세요.")
    st.stop()


# --------------------------------------------------
# 4. 결측값 처리
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
            "체질량지수의 결측값은 훈련 데이터 중앙값 "
            f"**{median_value:.2f}**으로 채웠습니다."
        )

if "bmi" not in selected_features:
    st.write(
        "체질량지수는 선택하지 않았으므로 BMI 결측값 대체를 하지 않았습니다."
    )


y_train = train_df["stroke"].astype(int).reset_index(drop=True)
y_test = test_df["stroke"].astype(int).reset_index(drop=True)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)


st.info(
    f"테스트용 사람 수: **{len(test_df):,}명** · "
    f"그중 실제 뇌졸중인 사람 수: "
    f"**{int((y_test == 1).sum()):,}명**"
)

st.caption(
    f"훈련 데이터: {len(train_df):,}명 · "
    f"테스트 데이터: {len(test_df):,}명"
)

if len(test_df) != 1533:
    st.warning(
        "테스트 데이터가 요청한 1,533명과 다릅니다. "
        "CSV의 행 수와 ID 구성을 확인해 주세요."
    )


# --------------------------------------------------
# 5. 두 모델 학습
# --------------------------------------------------

logistic_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=2000,
        random_state=42,
    ),
)

tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42,
)

try:
    logistic_model.fit(X_train, y_train)
    tree_model.fit(X_train, y_train)

except Exception as error:
    st.error("모델 학습 중 오류가 발생했습니다.")
    st.code(str(error))
    st.stop()


# --------------------------------------------------
# 6. 테스트 데이터 예측
# --------------------------------------------------

logistic_train_pred = logistic_model.predict(X_train)
logistic_test_pred = logistic_model.predict(X_test)

tree_train_pred = tree_model.predict(X_train)
tree_test_pred = tree_model.predict(X_test)


# 훈련 데이터에서 많은 쪽만 예측하는 기준 모델
majority_class = int(
    y_train.value_counts().idxmax()
)

baseline_train_pred = [
    majority_class
] * len(y_train)

baseline_test_pred = [
    majority_class
] * len(y_test)


# --------------------------------------------------
# 7. 혼동행렬 및 평가 지표 계산
# --------------------------------------------------

logistic_result = calculate_metrics(
    y_test,
    logistic_test_pred,
)

tree_result = calculate_metrics(
    y_test,
    tree_test_pred,
)

baseline_result = calculate_metrics(
    y_test,
    baseline_test_pred,
)


# --------------------------------------------------
# 8. 두 모델의 혼동행렬
# --------------------------------------------------

st.subheader("2. 모델별 혼동행렬")

st.write(
    "행은 실제 결과이고, 열은 모델의 예측 결과입니다. "
    "각 칸에는 해당하는 사람 수와 약자를 표시합니다."
)

left, right = st.columns(2)

with left:
    show_confusion_matrix(
        "로지스틱 회귀(확률로 답하는 모델)",
        logistic_result,
    )

with right:
    show_confusion_matrix(
        "의사결정트리(질문으로 답하는 모델)",
        tree_result,
    )


# --------------------------------------------------
# 9. 평가 지표 비교
# --------------------------------------------------

st.subheader("3. 평가 지표 비교")

metric_df = pd.DataFrame(
    [
        make_metric_row(
            "로지스틱 회귀(확률로 답하는 모델)",
            logistic_result,
        ),
        make_metric_row(
            "의사결정트리(질문으로 답하는 모델)",
            tree_result,
        ),
        make_metric_row(
            "훈련용에서 많은 쪽으로만 답하는 모델",
            baseline_result,
        ),
    ]
)

st.table(metric_df)

st.markdown(
    """
    **평가 지표의 의미**

    - 정확도: 전체 테스트 사람 중 정답을 맞힌 비율
    - 재현율: 실제 뇌졸중인 사람 중 뇌졸중이라고 맞힌 비율
    - 정밀도: 뇌졸중이라고 예측한 사람 중 실제 뇌졸중인 비율
    - F1: 정밀도와 재현율을 함께 고려한 값
    """
)

st.caption(
    "재현율 = TP / (TP + FN) · "
    "정밀도 = TP / (TP + FP) · "
    "F1 = 2 × 정밀도 × 재현율 / (정밀도 + 재현율)"
)

st.caption(
    "뇌졸중이라고 예측한 사람이 0명이면 "
    "정밀도와 F1은 '계산할 수 없음'으로 표시합니다."
)


# --------------------------------------------------
# 10. 놓친 환자와 헛짚은 사람 목록
# --------------------------------------------------

st.subheader("4. 잘못 예측한 사람 확인")

selected_model = st.selectbox(
    "목록을 확인할 모델을 선택하세요.",
    options=MODEL_NAMES,
)

if selected_model == MODEL_NAMES[0]:
    selected_predictions = logistic_test_pred
    selected_result = logistic_result
else:
    selected_predictions = tree_test_pred
    selected_result = tree_result


st.write(
    f"선택한 모델: **{selected_model}**"
)

# 목록을 만들 때 나이, 혈당, 고혈압, 심장병 여부를 사용
people_df = test_df[
    [
        "id",
        "age",
        "avg_glucose_level",
        "hypertension",
        "heart_disease",
    ]
].copy().reset_index(drop=True)

actual_values = y_test.reset_index(drop=True)
predicted_values = pd.Series(selected_predictions).reset_index(drop=True)

missed_df, missed_count = make_person_list(
    people_df,
    actual_values,
    predicted_values,
    "FN",
)

false_alarm_df, false_alarm_count = make_person_list(
    people_df,
    actual_values,
    predicted_values,
    "FP",
)


left, right = st.columns(2)

with left:
    st.markdown(
        f"#### 놓친 환자 (FN) · 전체 {missed_count:,}명"
    )

    st.caption(
        "실제로 뇌졸중이지만 뇌졸중이 아니라고 예측한 사람"
    )

    if missed_count == 0:
        st.success("놓친 환자가 없습니다.")
    else:
        st.dataframe(
            missed_df.head(20),
            use_container_width=True,
            hide_index=True,
        )

        if missed_count > 20:
            st.caption(
                f"전체 {missed_count:,}명 중 나이가 많은 "
                "앞의 20명만 표시했습니다."
            )


with right:
    st.markdown(
        f"#### 헛짚은 사람 (FP) · 전체 {false_alarm_count:,}명"
    )

    st.caption(
        "실제로는 뇌졸중이 아니지만 뇌졸중이라고 예측한 사람"
    )

    if false_alarm_count == 0:
        st.success("헛짚은 사람이 없습니다.")
    else:
        st.dataframe(
            false_alarm_df.head(20),
            use_container_width=True,
            hide_index=True,
        )

        if false_alarm_count > 20:
            st.caption(
                f"전체 {false_alarm_count:,}명 중 나이가 많은 "
                "앞의 20명만 표시했습니다."
            )


# --------------------------------------------------
# 11. 혼동행렬과 목록의 인원수 확인
# --------------------------------------------------

st.divider()

st.markdown("#### 목록과 혼동행렬의 인원수 확인")

st.write(
    f"놓친 환자 목록: **{missed_count:,}명** "
    f"= 혼동행렬 FN: **{selected_result['FN']:,}명**"
)

st.write(
    f"헛짚은 사람 목록: **{false_alarm_count:,}명** "
    f"= 혼동행렬 FP: **{selected_result['FP']:,}명**"
)

if (
    missed_count == selected_result["FN"]
    and false_alarm_count == selected_result["FP"]
):
    st.success(
        "두 목록의 인원수가 혼동행렬의 FN, FP와 일치합니다."
    )
else:
    st.error(
        "목록과 혼동행렬의 인원수가 일치하지 않습니다. "
        "데이터 정렬과 예측 결과를 확인해 주세요."
    )
