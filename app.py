import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import joblib
import pandas as pd

# ----------------------------------------------------------------------
# ตั้งค่าหน้าเว็บ
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="ระบบทำนายการรอดชีวิตผู้โดยสารไททานิก",
    page_icon="🚢",
    layout="centered",
)

# ----------------------------------------------------------------------
# ค่าคงที่ / ค่าที่ต้องแก้ไขเอง
# ----------------------------------------------------------------------
MODEL_PATH = "titanic_nb.joblib"

# TODO: แก้ไขค่านี้ให้ตรงกับผลการประเมิน (accuracy) จริงของโมเดลที่ฝึกไว้
MODEL_ACCURACY = 0.80

# ช่วงค่าที่ใช้ทำ Min-Max Normalization ตอนฝึกโมเดล (ค่ามาตรฐานของชุดข้อมูล Titanic)
AGE_MIN, AGE_MAX = 0.42, 80.0
FARE_MIN, FARE_MAX = 0.0, 512.3292

# ----------------------------------------------------------------------
# ธีมสีมินิมอล (Custom CSS)
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background-color: #F7F8FA;
        }
        .main-title {
            text-align: center;
            padding: 1.6rem 1rem 1rem 1rem;
        }
        .main-title h1 {
            font-size: 1.9rem;
            font-weight: 700;
            color: #1F2A44;
            margin-bottom: 0.2rem;
        }
        .main-title p {
            color: #6B7280;
            font-size: 0.95rem;
        }
        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 12px;
            padding: 0.8rem 1rem;
        }
        div[data-testid="stForm"] {
            background-color: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 1.4rem 1.4rem 0.6rem 1.4rem;
        }
        div.stButton > button, button[kind="formSubmit"] {
            background-color: #2F5D8C;
            color: #FFFFFF;
            border: none;
            border-radius: 8px;
            padding: 0.55rem 1.2rem;
            font-weight: 600;
            width: 100%;
        }
        div.stButton > button:hover, button[kind="formSubmit"]:hover {
            background-color: #244A70;
            color: #FFFFFF;
        }
        .footer {
            text-align: center;
            color: #9CA3AF;
            font-size: 0.85rem;
            margin-top: 2.5rem;
            padding-top: 1rem;
            border-top: 1px solid #E5E7EB;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# โหลดโมเดล
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ----------------------------------------------------------------------
# ส่วนหัวของเว็บ (ชื่อระบบ)
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="main-title">
        <h1>🚢 ระบบทำนายการรอดชีวิตผู้โดยสารไททานิก</h1>
        <p>Titanic Survival Prediction System · Gaussian Naive Bayes</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# แสดงค่าความแม่นยำของระบบ เพื่อสร้างความเชื่อมั่นให้ผู้ใช้
col1, col2, col3 = st.columns([1, 1.4, 1])
with col2:
    st.metric(label="ความแม่นยำของโมเดล (Accuracy)", value=f"{MODEL_ACCURACY * 100:.1f}%")

st.write("")
st.write("กรอกข้อมูลผู้โดยสารด้านล่าง แล้วกดปุ่ม **ทำนายผล**")

# ----------------------------------------------------------------------
# ฟอร์มกรอกข้อมูล
# ----------------------------------------------------------------------
def normalize(value, vmin, vmax):
    return (value - vmin) / (vmax - vmin)

with st.form("predict_form"):
    c1, c2 = st.columns(2)
    with c1:
        pclass = st.selectbox("ชั้นโดยสาร (Pclass)", options=[1, 2, 3], index=2)
        sex = st.radio("เพศ", options=["หญิง", "ชาย"], horizontal=True)
        age = st.slider("อายุ (ปี)", min_value=0, max_value=80, value=30)
    with c2:
        fare = st.number_input("ค่าโดยสาร (Fare)", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
        sibsp = st.number_input("จำนวนพี่น้อง/คู่สมรสที่ร่วมเดินทาง (SibSp)", min_value=0, max_value=10, value=0)
        parch = st.number_input("จำนวนพ่อแม่/ลูกที่ร่วมเดินทาง (Parch)", min_value=0, max_value=10, value=0)

    submitted = st.form_submit_button("ทำนายผล")

# ----------------------------------------------------------------------
# ทำนายผลและแสดงผลลัพธ์
# ----------------------------------------------------------------------
if submitted:
    sex_female = 1 if sex == "หญิง" else 0
    family_size = sibsp + parch + 1
    age_norm = normalize(age, AGE_MIN, AGE_MAX)
    fare_norm = normalize(fare, FARE_MIN, FARE_MAX)

    X = pd.DataFrame(
        [[pclass, sex_female, age_norm, fare_norm, family_size]],
        columns=["Pclass", "Sex_female", "Age", "Fare", "FamilySize"],
    )

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    st.write("")
    if pred == 1:
        st.success(f"🎉 ผลการทำนาย: **รอดชีวิต** (ความน่าจะเป็น {proba[1] * 100:.1f}%)")
    else:
        st.error(f"⚠️ ผลการทำนาย: **ไม่รอดชีวิต** (ความน่าจะเป็น {proba[0] * 100:.1f}%)")

    st.progress(float(proba[1]))
    st.caption(f"โอกาสรอดชีวิต {proba[1] * 100:.1f}% · โอกาสไม่รอดชีวิต {proba[0] * 100:.1f}%")

# ----------------------------------------------------------------------
# ท้ายหน้าเว็บ (ชื่อผู้พัฒนา)
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        พัฒนาโดย นายรชต ทัดทอง
    </div>
    """,
    unsafe_allow_html=True,
)
