import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import numpy as np
from sklearn.linear_model import LinearRegression

# 페이지 설정
st.set_page_config(
    page_title="Bitcoin Analysis & Prediction Dashboard",
    page_icon="₿",
    layout="wide"
)

# 데이터 로드 및 전처리 함수
@st.cache_data
def load_data(file_path):
    if not os.path.exists(file_path):
        return None
    
    # 1. CSV 로드 (제공된 파일의 구분자인 세미콜론 ';' 지정)
    df = pd.read_csv(file_path, sep=';')
    
    # 2. 날짜 컬럼 처리
    df['timeOpen'] = pd.to_datetime(df['timeOpen'])
    
    # 3. 데이터 정렬
    df = df.sort_values('timeOpen')
    
    # 4. 기술적 지표 추가
    df['SMA7'] = df['close'].rolling(window=7).mean()
    df['SMA30'] = df['close'].rolling(window=30).mean()
    
    return df

# 내일 가격 예측 함수 (선형 회귀)
def predict_next_day(df):
    # 날짜를 숫자로 변환 (학습을 위해)
    df_pred = df.copy()
    df_pred['date_ordinal'] = df_pred['timeOpen'].map(datetime.toordinal)
    
    # 특성(X)과 타겟(y) 설정
    X = df_pred[['date_ordinal']].values
    y = df_pred['close'].values
    
    # 모델 학습
    model = LinearRegression()
    model.fit(X, y)
    
    # 내일 날짜 계산
    next_day = df_pred['timeOpen'].max() + timedelta(days=1)
    next_day_ordinal = np.array([[next_day.toordinal()]])
    
    # 예측 수행
    prediction = model.predict(next_day_ordinal)[0]
    return next_day, prediction

# 메인 타이틀
st.title("₿ 비트코인 분석 및 내일 예측 대시보드")

CSV_FILE = "coin price.csv"
df = load_data(CSV_FILE)

if df is not None:
    # --- 예측 섹션 ---
    st.subheader("🔮 머신러닝 내일 가격 예측")
    next_date, predicted_price = predict_next_day(df)
    current_price = df.iloc[-1]['close']
    diff = predicted_price - current_price
    diff_pct = (diff / current_price) * 100

    pred_col1, pred_col2, pred_col3 = st.columns([1, 1, 2])
    
    with pred_col1:
        st.write(f"**예측 날짜:** {next_date.strftime('%Y-%m-%d')}")
        st.write(f"**예측 가격:** ₩{predicted_price:,.0f}")
        
    with pred_col2:
        if diff > 0:
            st.success(f"📈 **상승 예측** (+{diff_pct:.2f}%)")
        else:
            st.error(f"📉 **하락 예측** ({diff_pct:.2f}%)")
            
    with pred_col3:
        st.caption("※ 선형 회귀 모델 기반 예측으로, 실제 시장 상황과 다를 수 있습니다. 투자 참고용으로만 사용하세요.")

    st.divider()

    # --- 사이드바 및 필터 ---
    st.sidebar.header("📅 분석 기간 설정")
    min_date = df['timeOpen'].min().date()
    max_date = df['timeOpen'].max().date()
    
    try:
        date_range = st.sidebar.date_input(
            "조회할 날짜를 선택하세요",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    except Exception:
        date_range = [min_date, max_date]

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[(df['timeOpen'].dt.date >= start_date) & (df['timeOpen'].dt.date <= end_date)].copy()
    else:
        filtered_df = df.copy()

    # --- KPI 지표 ---
    if not filtered_df.empty:
        latest = filtered_df.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("현재 종가", f"₩{latest['close']:,.0f}")
        c2.metric("기간 최고가", f"₩{filtered_df['high'].max():,.0f}")
        c3.metric("기간 최저가", f"₩{filtered_df['low'].min():,.0f}")
        c4.metric("내일 예측가", f"₩{predicted_price:,.0f}", f"{diff_pct:+.2f}%")

        # --- 메인 차트 ---
        st.subheader("📈 시세 추이 및 예측 지점")
        fig_main = go.Figure()

        # 과거 데이터
        fig_main.add_trace(go.Scatter(
            x=filtered_df['timeOpen'], y=filtered_df['close'],
            mode='lines', name='과거 종가',
            line=dict(color='#F7931A', width=2)
        ))
        
        # 예측 데이터 점
        fig_main.add_trace(go.Scatter(
            x=[next_date], y=[predicted_price],
            mode='markers', name='내일 예측 지점',
            marker=dict(color='#FF4B4B', size=12, symbol='star')
        ))

        fig_main.update_layout(
            template="plotly_dark",
            xaxis_title="날짜",
            yaxis_title="가격 (KRW)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_main, use_container_width=True)

        # --- 상세 데이터 ---
        st.subheader("📊 일별 상세 데이터")
        st.dataframe(
            filtered_df[['timeOpen', 'open', 'high', 'low', 'close', 'volume']]
            .sort_values('timeOpen', ascending=False),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.warning("데이터가 없습니다.")

else:
    st.error(f"'{CSV_FILE}' 파일을 찾을 수 없습니다.")
