```python
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="분류 모델 | 뇌졸중 예측 실습실",
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


@st.cache_data(ttl=3600)
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


def show_accuracy_card(title, train_accuracy, test_accuracy):
    with st.container(border=True):
        st.markdown(f"#### {title}")
        st.metric("테스트 데이터 정확도", f"{test_accuracy:.2%}")

        left, right = st.columns(2)
        left.caption(f"훈련 데이터 정확도: **{train_accuracy:.2%}**")
        right.caption(f"테스트 데이터 정확도: **{test_accuracy:.2%}**")


st.title("🧠 분류 모델")
st.caption(
    "stroke=1은 뇌졸중(양성), stroke=0은 뇌졸중 아님(음성)으로 분류합니다."
)

# 데이터 불러오기
try:
    raw = load_data()
except Exception as error:
    st.error("데이터를 불러오지 못했습니다. GitHub 주소와 CSV 파일을 확인해 주세요.")
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
    col for col in required_columns
    if col not in raw.columns
]

if missing_columns:
    st.error("CSV 파일에 필요한 열이 없습니다: " + ", ".join(missing_columns))
    st.stop()


# ID 순으로 정렬한 뒤, 각 10명 묶음의 앞 3명을 테스트용으로 고정
data = raw.copy()

data["id"] = pd.to_numeric(data["id"], errors="coerce")
data["stroke"] = pd.to_numeric(data["stroke"], errors="coerce")

data = data.dropna(subset=["id", "stroke"])
data = data[data["stroke"].isin([0, 1])]
data = data.sort_values("id", kind="stable").reset_index(drop=True)

data["stroke"] = data["stroke"].astype(int)
data["_position"] = range(len(data))

test_mask = data["_position"] % 10 < 3

train_df = data.loc[~test_mask].copy()
test_df = data.loc[test_mask].copy()


# 1. 입력 속성 선택
st.subheader("1. 입력 속성 고르기")

selected_features = st.multiselect(
    "모델이 사용할 속성 (두 개 이상 선택)",
    options=list(FEATURE_NAMES.keys()),
    default=DEFAULT_FEATURES,
    format_func=lambda key: FEATURE_NAMES[key],
)

if len(selected_features) < 2:
    st.warning("입력 속성을 두 개 이상 골라 주세요.")
    st.stop()


if len(test_df) != 1533:
    st.warning(
        f"현재 테스트 데이터는 {len(test_df):,}명입니다. "
        "요청한 1,533명과 다르면 CSV의 행 수 또는 ID 구성을 확인해 주세요."
    )


# 선택된 입력 변수 숫자형 변환
for column in selected_features:
    train_df[column] = pd.to_numeric(
        train_df[column],
        errors="coerce",
    )
    test_df[column] = pd.to_numeric(
        test_df[column],
        errors="coerce",
    )


X_train = train_df[selected_features].copy()
X_test = test_df[selected_features].copy()

y_train = train_df["stroke"].copy()
y_test = test_df["stroke"].copy()


# BMI를 선택한 경우에만 훈련 데이터 중앙값으로 결측값 처리
if "bmi" in selected_features:
    imputer = SimpleImputer(strategy="median")

    X_train = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=selected_features,
        index=X_train.index,
    )

    X_test = pd.DataFrame(
        imputer.transform(X_test),
        columns=selected_features,
        index=X_test.index,
    )

    bmi_median = float(
        imputer.statistics_[selected_features.index("bmi")]
    )

    st.write(
        f"체질량지수의 빈 값은 훈련 데이터 중앙값 "
        f"**{bmi_median:.2f}**으로 채웠습니다."
    )

else:
    st.write(
        "체질량지수는 선택하지 않아 BMI 결측값 대체를 하지 않았습니다."
    )


# 선택된 다른 변수에 결측값이 있을 때도 훈련 데이터 중앙값 사용
for column in selected_features:
    median_value = X_train[column].median()

    if pd.isna(median_value):
        st.error(
            f"{FEATURE_NAMES[column]}의 훈련 데이터에 유효한 값이 없습니다."
        )
        st.stop()

    X_train[column] = X_train[column].fillna(median_value)
    X_test[column] = X_test[column].fillna(median_value)


st.info(
    f"훈련 데이터: {len(X_train):,}명 · "
    f"테스트 데이터: {len(X_test):,}명. "
    "ID 오름차순으로 정렬한 뒤 10명마다 앞 3명을 테스트용으로 분리했습니다."
)


# 2. 모델 학습
# 로지스틱 회귀: 표준화 후 학습
logistic_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=2000,
        random_state=42,
    ),
)

# 의사결정트리: 최대 깊이 3, 잎 노드 최소 인원 5명, 난수 고정
tree_model = DecisionTreeClassifier(
    max_depth=3,
    min_samples_leaf=5,
    random_state=42,
)

logistic_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)


# 입력을 보지 않고 훈련 데이터의 다수 클래스만 예측하는 기준 모델
majority_class = int(y_train.value_counts().idxmax())

baseline_train_predictions = [
    majority_class
] * len(y_train)

baseline_test_predictions = [
    majority_class
] * len(y_test)


# 모델별 훈련·테스트 정확도
results = [
    (
        "로지스틱 회귀(확률로 답하는 모델)",
        accuracy_score(
            y_train,
            logistic_model.predict(X_train),
        ),
        accuracy_score(
            y_test,
            logistic_model.predict(X_test),
        ),
    ),
    (
        "의사결정트리(질문으로 답하는 모델)",
        accuracy_score(
            y_train,
            tree_model.predict(X_train),
        ),
        accuracy_score(
            y_test,
            tree_model.predict(X_test),
        ),
    ),
    (
        "기준 모델(입력 없이 많은 쪽으로만 답하는 모델)",
        accuracy_score(
            y_train,
            baseline_train_predictions,
        ),
        accuracy_score(
            y_test,
            baseline_test_predictions,
        ),
    ),
]


# 3. 정확도 카드
st.subheader("2. 모델 정확도 비교")

cards = st.columns(3)

for card, (name, train_accuracy, test_accuracy) in zip(cards, results):
    with card:
        show_accuracy_card(
            name,
            train_accuracy,
            test_accuracy,
        )


st.caption(
    "기준 모델은 훈련 데이터에서 더 많은 쪽인 "
    + ("뇌졸중" if majority_class == 1 else "뇌졸중 아님")
    + "만 항상 예측합니다."
)


# 4. 산점도 축 선택
st.subheader("3. 테스트 데이터와 모델의 판단 영역")

left, right = st.columns(2)

with left:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        format_func=lambda key: FEATURE_NAMES[key],
    )

with right:
    y_options = [
        key for key in selected_features
        if key != x_feature
    ]

    y_feature = st.selectbox(
        "세로축 속성",
        options=y_options,
        index=0,
        format_func=lambda key: FEATURE_NAMES[key],
    )


# 두 축이 아닌 속성은 테스트 데이터 중앙값으로 고정
fixed_values = {
    key: float(X_test[key].median())
    for key in selected_features
    if key not in (x_feature, y_feature)
}

if fixed_values:
    fixed_text = " · ".join(
        f"{FEATURE_NAMES[key]} = {value:.3g}"
        for key, value in fixed_values.items()
    )

    st.write(
        "두 축이 아닌 속성은 테스트 데이터의 중앙값으로 고정했습니다: "
        + fixed_text
    )
else:
    st.write(
        "선택한 속성이 두 개뿐이므로 고정할 다른 속성은 없습니다."
    )


# 테스트 데이터 산점도
plot_df = X_test.copy()

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
    title="테스트 데이터의 실제 결과와 모델 판단 영역",
    opacity=0.7,
)


# 두 축의 판단 영역을 계산할 격자 생성
x_min = float(X_test[x_feature].min())
x_max = float(X_test[x_feature].max())

y_min = float(X_test[y_feature].min())
y_max = float(X_test[y_feature].max())

x_padding = (x_max - x_min) * 0.04 or 1.0
y_padding = (y_max - y_min) * 0.04 or 1.0

x_grid = [
    x_min - x_padding
    + i * (x_max - x_min + 2 * x_padding) / 100
    for i in range(101)
]

y_grid = [
    y_min - y_padding
    + i * (y_max - y_min + 2 * y_padding) / 100
    for i in range(101)
]


grid_rows = []

for y_value in y_grid:
    for x_value in x_grid:
        row = {}

        for feature in selected_features:
            if feature == x_feature:
                row[feature] = x_value
            elif feature == y_feature:
                row[feature] = y_value
            else:
                row[feature] = fixed_values[feature]

        grid_rows.append(row)


grid_df = pd.DataFrame(
    grid_rows,
   
