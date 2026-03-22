import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="ทำนายราคารถ BMW",
    page_icon="🚗",
    layout="wide"
)

# Path ไฟล์
BASE_DIR = Path(__file__).parent
ARTIFACTS_DIR = BASE_DIR / "model_artifacts"

MODEL_PATH = ARTIFACTS_DIR / "bmw_price_pipeline.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "feature_names.json"
METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

# โหลดโมเดล (กันพัง)
@st.cache_resource
def load_model():
    try:
        pipeline = joblib.load(MODEL_PATH)

        with open(FEATURES_PATH, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return pipeline, feature_names, metadata

    except Exception as e:
        st.error(f"❌ โหลดโมเดลไม่สำเร็จ: {e}")
        return None, None, None

pipeline, feature_names, metadata = load_model()

if pipeline is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.header("📊 ข้อมูลโมเดล")

    st.write(f"โมเดล: {metadata.get('selected_model', '-')}")
    st.write(f"ประเภท: {metadata.get('problem_type', '-')}")
    st.write(f"ตัวแปรเป้าหมาย: {metadata.get('target', '-')}")

# หัวข้อ
st.title("🚗 ระบบทำนายราคารถ BMW")
st.write("กรอกข้อมูลรถของคุณเพื่อประเมินราคา")

# Input
col1, col2 = st.columns(2)
input_data = {}

with col1:
    input_data["model"] = st.selectbox("รุ่นรถ", ["3 Series", "5 Series", "7 Series", "X1", "X3", "X5", "X7", "อื่นๆ"])
    input_data["year"] = st.slider("ปีรถ", 1985, 2026, 2018)
    input_data["engine_size"] = st.number_input("ขนาดเครื่องยนต์ (ลิตร)", 0.8, 6.0, 2.0)
    input_data["horsepower"] = st.number_input("แรงม้า", 60, 800, 180)
    input_data["fuel_type"] = st.selectbox("ประเภทเชื้อเพลิง", ["Petrol", "Diesel", "Hybrid", "Electric"])
    input_data["transmission"] = st.selectbox("เกียร์", ["Automatic", "Manual"])

with col2:
    input_data["drivetrain"] = st.selectbox("ระบบขับเคลื่อน", ["RWD", "AWD", "FWD"])
    input_data["mileage_km"] = st.slider("ระยะทาง (กม.)", 0, 300000, 80000)
    input_data["doors"] = st.selectbox("จำนวนประตู", [2, 4, 5])
    input_data["seats"] = st.selectbox("จำนวนที่นั่ง", [2, 4, 5, 7])
    input_data["body_type"] = st.selectbox("ประเภทรถ", ["Sedan", "SUV", "Coupe", "Hatchback", "Wagon"])
    input_data["owner_count"] = st.slider("จำนวนเจ้าของ", 0, 5, 1)
    input_data["accident_history"] = st.selectbox("เคยชน", [0, 1], format_func=lambda x: "ไม่เคย" if x == 0 else "เคย")
    input_data["service_history"] = st.selectbox("มีประวัติเข้าศูนย์", [0, 1], format_func=lambda x: "ไม่มี" if x == 0 else "มี")

# ปุ่มทำนาย
if st.button("🔮 ทำนายราคา"):

    try:
        df = pd.DataFrame([input_data])

        # 🔥 เติม feature ที่ขาด
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0

        # เรียง column ให้ตรง model
        df = df[feature_names]

        price = pipeline.predict(df)[0]

        st.success(f"💰 ราคาประมาณ: ${price:,.0f}")

        with st.expander("📋 ดูข้อมูลที่ใช้ทำนาย"):
            st.dataframe(df)

    except Exception as e:
        st.error(f"❌ ทำนายไม่สำเร็จ: {e}")