import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 페이지 기본 설정
st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")

# 1. 데이터 불러오기
df = pd.read_csv("data/Heart_Disease_Prediction.csv")

# 2. 결측치 및 이상치 처리 (간단히 결측치만 제거)
df.dropna(inplace=True)

# 타겟 컬럼 인코딩 (Presence/Absence → 1/0)
df['Heart Disease'] = df['Heart Disease'].map({'Presence': 1, 'Absence': 0})

# 3. 정규화 및 표준화
X = df.drop(columns=["Heart Disease"])
y = df["Heart Disease"]

# 범주형 인코딩
for col in X.select_dtypes("object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# 정규화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train-Test 분리
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 6. 예측 및 성능 평가
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

conf_matrix = confusion_matrix(y_test, y_pred)

# 변수 중요도
feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  사이드바 메뉴
#  → 각 옵션에서 불필요한 공백 제거
menu = st.sidebar.radio("메뉴 선택", ["Home", "데이터분석", "데이터시각화", "머신러닝보고서"])
st.header(menu)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  HOME
if menu == "Home":
    st.title("심장병 데이터 개요")
    st.markdown("""
    - **데이터 출처**: [Kaggle - Heart Disease Dataset](https://www.kaggle.com/)
    - **데이터 설명**: 심장 질환 유무(Presence/Absence)와 관련된 생리학적 특성들을 포함한 데이터셋입니다.
    - **컬럼 예시**:
        - `Age`: 나이
        - `Sex`: 성별
        - `ChestPainType`: 가슴통증 유형
        - `BP`: 혈압
        - `Cholesterol`: 콜레스테롤 수치
        - `MaxHR`: 최대 심박수
        - `ExerciseAngina`: 운동 중 협심증 여부
        - `ST_Slope`: ST segment 경사
    """)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 데이터 분석
elif menu == "데이터분석":
    st.title("심장병 데이터 분석")

    tab1, tab2, tab3 = st.tabs(["상위데이터", "데이터통계", "조건데이터"])

    with tab1:
        st.subheader("데이터 상위 10개")
        st.dataframe(df.head(10))

    with tab2:
        st.subheader("데이터 통계 요약")
        st.dataframe(df.describe())

    with tab3:
        st.subheader("조건 검색: 나이 60세 이상")
        st.dataframe(df[df["Age"] >= 60])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 데이터 시각화
elif menu == "데이터시각화":
    st.title("데이터 시각화")

    tab1, tab2, tab3 = st.tabs(["히스토그램", "박스플롯", "히트맵"])

    with tab1:
        st.subheader("나이 분포")
        fig1, ax1 = plt.subplots()
        sns.histplot(df["Age"], kde=True, bins=20, ax=ax1)
        st.pyplot(fig1)

    with tab2:
        st.subheader("성별에 따른 MaxHR 분포 (Heart Disease 여부)")
        fig2, ax2 = plt.subplots()
        sns.boxplot(data=df, x="Sex", y="Max HR", hue="Heart Disease", palette="Set2", ax=ax2)
        ax2.set_title("MaxHR by Gender & Heart Disease")
        st.pyplot(fig2)

        st.subheader("성별에 따른 ST depression 분포 (Heart Disease 여부)")
        fig3, ax3 = plt.subplots()
        sns.boxplot(data=df, x="Sex", y="ST depression", hue="Heart Disease", palette="Set2", ax=ax3)
        ax3.set_title("ST depression by Gender & Heart Disease")  # ax3.set_title로 수정
        st.pyplot(fig3)

    with tab3:
        st.subheader("상관관계 히트맵")
        numeric_df = df.select_dtypes(include='number')
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax4)
        st.pyplot(fig4)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 머신러닝 보고서
elif menu == "머신러닝보고서":
    st.title("심장병 머신러닝 보고서")

    st.metric("정확도 (Accuracy)", f"{accuracy:.2%}")

    st.subheader("Classification Report")
    st.dataframe(report_df.round(2))

    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    st.subheader("입력 변수 중요도")
    fig_imp, ax_imp = plt.subplots(figsize=(8, 6))
    sns.barplot(data=feature_importances, y="Feature", x="Importance", palette="viridis", ax=ax_imp)
    ax_imp.set_title("Feature Importance")
    st.pyplot(fig_imp)
