
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

    if "bmi"
