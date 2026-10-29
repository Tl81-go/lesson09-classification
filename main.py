import json
from pathlib import Path

import pandas as pd
import streamlit as st


# 페이지 설정
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 뇌졸중 예측 실습실")
st.write("뇌졸중 데이터의 구조와 특징을 살펴보는 실습 공간입니다.")

# 데이터 주소
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "Tl81-go/data/refs/heads/main/stroke.csv"
)

# 기록 파일 경로
RECORD_DIR = Path("내_기록")
RECORD_FILE = RECORD_DIR / "stroke_dictionary.json"


# 데이터 불러오기
@st.cache_data(ttl=3600)
def load_data():
    return pd.read_csv(DATA_URL, encoding="utf-8")


# 저장된 기록 불러오기
def load_notes():
    try:
        if RECORD_FILE.exists():
            with RECORD_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)

            return {
                "meanings": data.get("meanings", {}),
                "source": data.get("source", ""),
            }
    except (OSError, json.JSONDecodeError):
        pass

    return {"meanings": {}, "source": ""}


# 기록 저장하기
def save_notes(meanings, source):
    RECORD_DIR.mkdir(parents=True, exist_ok=True)

    data = {
        "meanings": meanings,
        "source": source,
    }

    with RECORD_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# 데이터 읽기
try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.write("GitHub 데이터 주소와 파일 공개 여부를 확인해 주세요.")
    st.code(str(e))
    st.stop()

if df.empty:
    st.error("데이터가 비어 있습니다.")
    st.stop()

# 기록 읽기
notes = load_notes()
meanings = notes["meanings"]
saved_source = notes["source"]

# 뇌졸중 기록 수 계산
if "stroke" in df.columns:
    stroke_values = pd.to_numeric(df["stroke"], errors="coerce")
    stroke_count = int((stroke_values == 1).sum())
else:
    stroke_count = 0

stroke_ratio = stroke_count / len(df) * 100

# 숫자 카드
st.subheader("📊 데이터 한눈에 보기")

col1, col2, col3, col4 = st.columns(4)

col1.metric("전체 사람 수", f"{len(df):,}명")
col2.metric("열 개수", f"{len(df.columns)}개")
col3.metric("stroke가 1인 사람 수", f"{stroke_count:,}명")
col4.metric("stroke가 1인 비율", f"{stroke_ratio:.2f}%")

st.divider()

# 열 정보 표
st.subheader("🗂️ 열 정보와 우리말 뜻")
st.write("우리말 뜻을 직접 입력한 뒤 저장 버튼을 눌러 주세요.")

rows = []

for column in df.columns:
    series = df[column]
    unique_count = series.nunique(dropna=True)

    if pd.api.types.is_numeric_dtype(series):
        kind = f"숫자형 (고유값 {unique_count}개)"
    else:
        kind = f"문자형/범주형 (고유값 {unique_count}개)"

    rows.append({
        "열 이름": column,
        "우리말 뜻": meanings.get(column, ""),
        "값의 종류": kind,
        "빈 값 개수": int(series.isna().sum()),
    })

info_df = pd.DataFrame(rows)

edited_df = st.data_editor(
    info_df,
    use_container_width=True,
    hide_index=True,
    disabled=[
        "열 이름",
        "값의 종류",
        "빈 값 개수",
    ],
    column_config={
        "열 이름": st.column_config.TextColumn("열 이름"),
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻",
            help="교재를 보고 직접 입력하세요.",
        ),
        "값의 종류": st.column_config.TextColumn("값의 종류"),
        "빈 값 개수": st.column_config.NumberColumn("빈 값 개수"),
    },
    key="meaning_editor",
)

st.divider()

# 원본 데이터 처음 다섯 줄
st.subheader("🔎 데이터 처음 다섯 줄")
st.dataframe(
    df.head(5),
    use_container_width=True,
    hide_index=True,
)

st.divider()

# 데이터 출처 입력
st.subheader("📚 데이터 출처")

source = st.text_area(
    "교재에 적힌 데이터 출처를 입력하세요.",
    value=saved_source,
    height=100,
)

# 저장 버튼
if st.button("💾 우리말 뜻과 출처 저장", type="primary"):

    new_meanings = {}

    for _, row in edited_df.iterrows():
        value = row["우리말 뜻"]

        if pd.isna(value):
            value = ""

        new_meanings[str(row["열 이름"])] = str(value).strip()

    try:
        save_notes(new_meanings, source.strip())
        st.success("우리말 뜻과 데이터 출처를 저장했습니다.")
        st.caption(f"저장 위치: {RECORD_FILE}")
    except OSError as e:
        st.error("파일 저장에 실패했습니다.")
        st.code(str(e))

if source.strip():
    st.write("현재 입력한 출처:")
    st.write(source.strip())

st.caption(
    "클라우드 환경에서는 재배포 후 로컬 JSON 파일이 "
    "유지되지 않을 수 있습니다."
)
