
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz


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
        left.caption(f"훈련 정확도: {train_accuracy:.2%}")
        right.caption(f"테스트 정확도: {test_accuracy:.2%}")


def prepare_features(train_df, test_df, features):
    """훈련 데이터의 중앙값으로 결측값을 처리한다."""
    X_train = train_df[features].copy()
    X_test = test_df[features].copy()

    for feature in features:
        X_train[feature] = pd.to_numeric(
            X_train[feature], errors="coerce"
        )
        X_test[feature] = pd.to_numeric(
            X_test[feature], errors="coerce"
        )

        median_value = X_train[feature].median()

        if pd.isna(median_value):
            raise ValueError(
                f"{FEATURE_NAMES[feature]}의 훈련 데이터에 "
                "유효한 숫자가 없습니다."
            )

        # BMI는 선택된 경우에만 결측값을 대체한다.
        # 다른 선택 변수에도 결측값이 있다면 해당 변수의 중앙값을 사용한다.
        X_train[feature] = X_train[feature].fillna(median_value)
        X_test[feature] = X_test[feature].fillna(median_value)

        if feature == "bmi":
            st.write(
                "체질량지수의 빈 값은 훈련 데이터 중앙값 "
                f"**{median_value:.2f}**으로 채웠습니다."
            )

    if "bmi" not in features:
        st.write(
            "체질량지수는 선택하지 않았으므로 BMI 결측값을 "
            "대체하지 않았습니다."
        )

    return X_train, X_test


st.title("🧠 분류 모델")
st.caption(
    "stroke=1은 뇌졸중(양성), stroke=0은 뇌졸중 아님(음성)입니다."
)

# --------------------------------------------------
# 1. 데이터 불러오기 및 분리
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


data = raw.copy()

data["id"] = pd.to_numeric(data["id"], errors="coerce")
data["stroke"] = pd.to_numeric(data["stroke"], errors="coerce")

data = data.dropna(subset=["id", "stroke"])
data = data[data["stroke"].isin([0, 1])]
data = data.sort_values("id", kind="stable").reset_index(drop=True)

data["stroke"] = data["stroke"].astype(int)

# ID 순으로 정렬한 뒤 10명씩 묶어 앞의 3명을 테스트용으로 사용
data["_position"] = np.arange(len(data))
test_mask = data["_position"] % 10 < 3

train_df = data.loc[~test_mask].copy()
test_df = data.loc[test_mask].copy()

if train_df.empty or test_df.empty:
    st.error("훈련 또는 테스트 데이터가 비어 있습니다.")
    st.stop()

if len(test_df) != 1533:
    st.warning(
        f"현재 테스트 데이터는 {len(test_df):,}명입니다. "
        "요청한 1,533명과 다르면 CSV 행 수와 ID 구성을 확인해 주세요."
    )


# --------------------------------------------------
# 2. 입력 속성 선택
# --------------------------------------------------

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


try:
    X_train, X_test = prepare_features(
        train_df,
        test_df,
        selected_features,
    )
except ValueError as error:
    st.error(str(error))
    st.stop()

y_train = train_df["stroke"].astype(int)
y_test = test_df["stroke"].astype(int)

st.info(
    f"훈련 데이터: {len(X_train):,}명 · "
    f"테스트 데이터: {len(X_test):,}명"
    "\n\nID 오름차순으로 정렬한 뒤 10명마다 앞의 3명을 "
    "테스트 데이터로 분리했습니다."
)


# --------------------------------------------------
# 3. 모델 학습
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


# 훈련 데이터에서 더 많은 클래스를 예측하는 기준 모델
majority_class = int(y_train.value_counts().idxmax())

baseline_train_predictions = np.full(
    len(y_train), majority_class
)
baseline_test_predictions = np.full(
    len(y_test), majority_class
)


results = [
    (
        "로지스틱 회귀(확률로 답하는 모델)",
        accuracy_score(y_train, logistic_model.predict(X_train)),
        accuracy_score(y_test, logistic_model.predict(X_test)),
    ),
    (
        "의사결정트리(질문으로 답하는 모델)",
        accuracy_score(y_train, tree_model.predict(X_train)),
        accuracy_score(y_test, tree_model.predict(X_test)),
    ),
    (
        "기준 모델(많은 쪽으로만 답하는 모델)",
        accuracy_score(y_train, baseline_train_predictions),
        accuracy_score(y_test, baseline_test_predictions),
    ),
]


# --------------------------------------------------
# 4. 정확도 카드
# --------------------------------------------------

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
    + (
        "뇌졸중"
        if majority_class == 1
        else "뇌졸중 아님"
    )
    + "만 항상 예측합니다."
)


# --------------------------------------------------
# 5. 테스트 데이터 산점도와 판단 영역
# --------------------------------------------------

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


fixed_values = {
    feature: float(X_test[feature].median())
    for feature in selected_features
    if feature not in (x_feature, y_feature)
}

if fixed_values:
    fixed_text = " · ".join(
        f"{FEATURE_NAMES[feature]} = {value:.3g}"
        for feature, value in fixed_values.items()
    )
    st.write(
        "두 축이 아닌 속성은 테스트 데이터의 중앙값으로 고정했습니다: "
        + fixed_text
    )
else:
    st.write("두 축 외에 고정할 다른 속성은 없습니다.")


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
        ]
    },
    labels={
        x_feature: FEATURE_NAMES[x_feature],
        y_feature: FEATURE_NAMES[y_feature],
    },
    title="테스트 데이터의 실제 결과와 모델 판단 영역",
    opacity=0.7,
)


# 축 범위 설정
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

x_padding = (x_max - x_min) * 0.04
y_padding = (y_max - y_min) * 0.04

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
    columns=selected_features,
)


# 의사결정트리 판단 영역을 옅게 표시
tree_grid = tree_model.predict(grid_df).to_numpy().reshape(
    len(y_grid), len(x_grid)
)

fig.add_trace(
    go.Contour(
        x=x_grid,
        y=y_grid,
        z=tree_grid,
        showscale=False,
        opacity=0.16,
        hoverinfo="skip",
        contours={
            "start": 0,
            "end": 1,
            "size": 1,
            "coloring": "fill",
            "showlines": False,
        },
        colorscale=[
            [0, "lightblue"],
            [1, "lightpink"],
        ],
        name="의사결정트리 영역",
    )
)


# 로지스틱 회귀의 확률 0.5 경계선
probabilities = logistic_model.predict_proba(grid_df)[:, 1]
probability_grid = probabilities.reshape(len(y_grid), len(x_grid))

fig.add_trace(
    go.Contour(
        x=x_grid,
        y=y_grid,
        z=probability_grid,
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
            "color": "black",
            "width": 3,
        },
        name="로지스틱 회귀 확률 0.5 경계",
    )
)

fig.update_layout(
    height=650,
    legend_title_text="실제 결과",
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "옅은 배경은 의사결정트리의 예측 영역이고, "
    "검은 선은 로지스틱 회귀가 뇌졸중 확률 50%를 판단하는 경계입니다."
)


# --------------------------------------------------
# 6. 의사결정트리 구조
# --------------------------------------------------

st.subheader("4. 의사결정트리 구조")

dot = export_graphviz(
    tree_model,
    out_file=None,
    feature_names=[
        FEATURE_NAMES[feature]
        for feature in selected_features
    ],
    class_names=[
        "뇌졸중 아님",
        "뇌졸중",
    ],
    filled=True,
    rounded=True,
    special_characters=False,
    impurity=False,
    proportion=False,
    node_ids=True,
)

# 각 노드에 훈련 인원, 실제 뇌졸중 양성 인원, 양성 비율을 표시
tree_structure = tree_model.tree_
dot_lines = dot.splitlines()
updated_lines = []

for line in dot_lines:
    if " [label=" in line:
        try:
            node_text = line.split(" [label=")[0].strip()
            node_id = int(node_text)

            counts = tree_structure.value[node_id][0]
            total_count = int(tree_structure.n_node_samples[node_id])

            if len(counts) > 1:
                positive_count = int(counts[1])
            else:
                positive_count = 0

            positive_ratio = (
                positive_count / total_count
                if total_count > 0
                else 0
            )

            extra = (
                f"\\n훈련 인원 = {total_count}"
                f"\\n뇌졸중 양성 = {positive_count}"
                f"\\n양성 비율 = {positive_ratio:.1%}"
            )

            line = line.replace(
                '",',
                extra + '",',
                1,
            )

        except (ValueError, IndexError):
            pass

    updated_lines.append(line)

dot = "\n".join(updated_lines)

# 잎 노드는 예측 클래스에 따라 색상을 지정
leaf_ids = [
    node_id
    for node_id in range(tree_structure.node_count)
    if tree_structure.children_left[node_id]
    == tree_structure.children_right[node_id]
]

leaf_color_lines = []

for node_id in leaf_ids:
    class_counts = tree_structure.value[node_id][0]
    predicted_class = int(np.argmax(class_counts))

    color = "#D6EAF8" if predicted_class == 0 else "#FADBD8"

    leaf_color_lines.append(
        f'{node_id} [fillcolor="{color}", style="filled,rounded"];'
    )

dot = dot.replace(
    "\n}",
    "\n" + "\n".join(leaf_color_lines) + "\n}",
)

# 가지에 예/아니요 표시
dot = dot.replace(
    'label="True"',
    'label="예"',
).replace(
    'label="False"',
    'label="아니요"',
)

st.graphviz_chart(dot, use_container_width=True)


# 잎 노드 통계
leaf_predictions = []

for node_id in leaf_ids:
    class_counts = tree_structure.value[node_id][0]
    predicted_class = int(np.argmax(class_counts))
    leaf_predictions.append(predicted_class)

no_stroke_leaves = sum(
    prediction == 0
    for prediction in leaf_predictions
)

st.write(f"**전체 잎 노드 수:** {len(leaf_ids)}개")
st.write(
    f"**뇌졸중 아님(0)으로 예측하는 잎 노드:** "
    f"{no_stroke_leaves}개"
)

# 실제 분기에서 사용된 변수
used_feature_indices = sorted({
    int(tree_structure.feature[node_id])
    for node_id in range(tree_structure.node_count)
    if tree_structure.children_left[node_id]
    != tree_structure.children_right[node_id]
})

if used_feature_indices:
    used_features = [
        FEATURE_NAMES[selected_features[index]]
        for index in used_feature_indices
    ]
    st.write(
        "**트리가 실제로 사용한 분기 속성:** "
        + ", ".join(used_features)
    )
else:
    st.write("트리가 분기에 사용한 속성이 없습니다.")
