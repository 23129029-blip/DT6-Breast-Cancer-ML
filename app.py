import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.datasets import load_breast_cancer

st.set_page_config(page_title="BreastCare AI", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.stApp{background:#F3F7FB;color:#19364B}
[data-testid="stHeader"]{background:#F3F7FB}
.block-container{max-width:1450px;padding-top:1.5rem}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#102F49,#174C65)}
[data-testid="stSidebar"] *{color:#fff!important}
h1,h2,h3,h4{color:#173B57!important;font-weight:800!important}
p,label{color:#29465B}
.hero{background:linear-gradient(120deg,#102F49,#087F8C);padding:28px;border-radius:18px;margin-bottom:20px}
.hero h1{color:#fff!important;font-size:29px}
.hero p{color:#EAFBFC!important;line-height:1.8}
.metric-card{background:#fff;border:1px solid #DCE7EF;border-radius:14px;padding:18px;min-height:115px}
.metric-label{color:#61778A;font-size:12px;font-weight:750;margin-bottom:10px}
.metric-value{color:#123F5A;font-size:25px;font-weight:800}
.metric-note{color:#087F8C;font-size:12px;margin-top:8px}
[data-testid="stMetric"]{background:#fff;border:1px solid #DCE7EF;border-radius:13px;padding:15px}
[data-testid="stMetricLabel"]{color:#526B7D!important}
[data-testid="stMetricValue"]{color:#123F5A!important}
[data-baseweb="select"]>div,[data-baseweb="input"]>div{background:#fff;border:1px solid #C9D8E3;border-radius:8px}
input{color:#173B57!important}
div.stButton>button,div[data-testid="stFormSubmitButton"] button,div[data-testid="stDownloadButton"] button{background:#087F8C;color:#fff!important;border:0;border-radius:9px;min-height:44px;font-weight:750}
div.stButton>button:hover,div[data-testid="stFormSubmitButton"] button:hover{background:#075D69;color:#fff!important}
.result-box{padding:22px;border-radius:14px;margin:14px 0;text-align:center}
.result-malignant{background:#FFF0EF;border:1px solid #F4B8B2;color:#9F2018}
.result-benign{background:#EAF8F0;border:1px solid #A9DCC0;color:#176B45}
.result-title{font-size:22px;font-weight:800;margin-bottom:8px}
.result-subtitle{font-size:14px;line-height:1.8}
.footer{color:#627789;text-align:center;font-size:12px;line-height:1.9;border-top:1px solid #DCE7EF;padding-top:20px;margin-top:30px}
</style>
""", unsafe_allow_html=True)


MODEL_PATH = Path(__file__).resolve().parent / "svm_rbf_final.joblib"


@st.cache_resource
def load_artifact(path):
    return joblib.load(path)

if not MODEL_PATH.exists():
    st.error("Không tìm thấy model_final/svm_rbf_final.joblib. Hãy chạy app.py từ thư mục DT6_Breast_Cancer_ML.")
    st.stop()

artifact = load_artifact(str(MODEL_PATH))
model = artifact["model"]
threshold = float(artifact["threshold"])
best_params = artifact.get("best_params", {})
cv_auc = artifact.get("cv_best_auc")

data = load_breast_cancer()
features = list(data.feature_names)
df = pd.DataFrame(data.data, columns=features)
# Quy uoc cua du an: 1 = malignant, 0 = benign
df["diagnosis"] = 1 - data.target

if hasattr(model, "named_steps") and "select" in model.named_steps:
    mask = model.named_steps["select"].get_support()
    selected_features = [features[i] for i, keep in enumerate(mask) if keep]
else:
    selected_features = artifact.get("selected_features", features)

feature_vi = {
    "mean radius":"Bán kính trung bình","mean texture":"Kết cấu trung bình",
    "mean perimeter":"Chu vi trung bình","mean area":"Diện tích trung bình",
    "mean smoothness":"Độ trơn trung bình","mean compactness":"Độ đặc trưng trung bình",
    "mean concavity":"Độ lõm trung bình","mean concave points":"Điểm lõm trung bình",
    "mean symmetry":"Độ đối xứng trung bình","mean fractal dimension":"Chiều fractal trung bình",
    "radius error":"Sai số bán kính","texture error":"Sai số kết cấu",
    "perimeter error":"Sai số chu vi","area error":"Sai số diện tích",
    "smoothness error":"Sai số độ trơn","compactness error":"Sai số độ đặc trưng",
    "concavity error":"Sai số độ lõm","concave points error":"Sai số điểm lõm",
    "symmetry error":"Sai số độ đối xứng","fractal dimension error":"Sai số chiều fractal",
    "worst radius":"Bán kính lớn nhất","worst texture":"Kết cấu lớn nhất",
    "worst perimeter":"Chu vi lớn nhất","worst area":"Diện tích lớn nhất",
    "worst smoothness":"Độ trơn lớn nhất","worst compactness":"Độ đặc trưng lớn nhất",
    "worst concavity":"Độ lõm lớn nhất","worst concave points":"Điểm lõm lớn nhất",
    "worst symmetry":"Độ đối xứng lớn nhất","worst fractal dimension":"Chiều fractal lớn nhất"
}

def malignant_probability(frame):
    probs = model.predict_proba(frame)[0]
    return float(probs[list(model.classes_).index(1)])

# SIDEBAR: ten danh muc khong co sticker; radio co chuc nang dieu huong that
with st.sidebar:
    st.markdown("## BreastCare AI")
    st.markdown("### Hệ thống hỗ trợ nghiên cứu")
    st.markdown("---")
    page = st.radio("DANH MỤC", ["Tổng quan", "Phân loại khối u", "Đánh giá mô hình"], key="page_navigation")
    st.markdown("---")
    st.markdown("### Cấu hình mô hình")
    st.write("**Thuật toán:** SVM RBF")
    st.write(f"**Đặc trưng:** {len(selected_features)}/30")
    st.write(f"**Ngưỡng:** {threshold:.5f}")
    with st.expander("Xem đặc trưng được chọn"):
        for f in selected_features:
            st.write("• " + f)
    st.markdown("---")
    st.caption("Phiên bản học tập và nghiên cứu")

st.markdown("""
<div class="hero">
<h1>Hệ thống hỗ trợ phân loại khối u vú bằng máy học</h1>
<p>BREASTCARE AI · SVM RBF · BREAST CANCER WISCONSIN DATASET<br>
Công cụ hỗ trợ nghiên cứu và minh họa quy trình phân loại.</p>
</div>
""", unsafe_allow_html=True)
st.info("Ứng dụng phục vụ mục đích học tập và nghiên cứu, không thay thế việc thăm khám hoặc chẩn đoán y khoa.")

if page == "Tổng quan":
    st.markdown("## Tổng quan hệ thống")
    st.caption("Thông tin mô hình cuối đang được nạp trong ứng dụng.")
    a,b,c,d = st.columns(4)
    cards = [
        ("THUẬT TOÁN","SVM RBF","Phân loại nhị phân"),
        ("ĐẶC TRƯNG",f"{len(selected_features)} / 30","SelectKBest"),
        ("NGƯỠNG PHÂN LOẠI",f"{threshold:.5f}","Đọc từ mô hình đã lưu"),
        ("AUC CV HUẤN LUYỆN",f"{float(cv_auc)*100:.2f}%" if cv_auc is not None else "Chưa có","Không phải kiểm định độc lập")
    ]
    for col,(label,value,note) in zip([a,b,c,d],cards):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',unsafe_allow_html=True)
    st.markdown("---")
    left,right = st.columns(2)
    with left:
        st.markdown("### Phân loại khối u")
        st.write("Chọn mẫu hoặc nhập thông số, sau đó nhấn nút dự đoán.")
    with right:
        st.markdown("### Đánh giá mô hình")
        st.write("Kiểm tra 569 mẫu, xem Accuracy, ma trận nhầm lẫn và các mẫu dự đoán sai.")
    if best_params:
        with st.expander("Xem tham số tốt nhất"):
            st.json(best_params)

elif page == "Phân loại khối u":
    st.markdown("## Phân tích mẫu đầu vào")
    st.caption("Chọn mẫu có sẵn hoặc nhập thông số đo.")
    option = st.selectbox("Phương thức nhập dữ liệu",[
        "Nhập dữ liệu thủ công","Dữ liệu mẫu - Malignant","Dữ liệu mẫu - Benign","Chọn mẫu từ Dataset"
    ])
    sample = {}
    if option == "Chọn mẫu từ Dataset":
        sample_index = st.number_input("Số thứ tự mẫu (1–569)",1,len(df),1,1)
        row = df.iloc[int(sample_index)-1]
        sample = {f:float(row[f]) for f in features}
        st.warning("Nhãn thực tế: " + ("Malignant (ác tính)" if int(row["diagnosis"])==1 else "Benign (lành tính)"))
    elif option == "Dữ liệu mẫu - Malignant":
        row = df[df.diagnosis==1].iloc[0]
        sample = {f:float(row[f]) for f in features}
    elif option == "Dữ liệu mẫu - Benign":
        row = df[df.diagnosis==0].iloc[0]
        sample = {f:float(row[f]) for f in features}

    with st.expander("Mô hình sử dụng những đặc trưng nào?"):
        st.write(f"Mô hình chọn {len(selected_features)} đặc trưng trong 30 đặc trưng.")
        st.write(", ".join(selected_features))

    st.markdown("### Thông số đầu vào")
    st.caption("Giá trị mặc định là trung vị trong dataset khi nhập thủ công; có thể điều chỉnh.")
    with st.form("prediction_form"):
        inputs = {}
        cols = st.columns(3)
        for i,f in enumerate(features):
            value = sample[f] if option != "Nhập dữ liệu thủ công" else float(df[f].median())
            with cols[i%3]:
                inputs[f] = st.number_input(feature_vi.get(f,f), min_value=0.0, value=value, format="%.6f", key=f"input_{f}")
                st.caption(f)
        submitted = st.form_submit_button("PHÂN TÍCH VÀ DỰ ĐOÁN", use_container_width=True)

    if submitted:
        frame = pd.DataFrame([inputs], columns=features)
        p_m = malignant_probability(frame)
        p_b = 1.0-p_m
        pred = int(p_m >= threshold)
        st.markdown("### Kết quả dự đoán")
        if pred == 1:
            st.markdown('<div class="result-box result-malignant"><div class="result-title">MALIGNANT — ÁC TÍNH</div><div class="result-subtitle">Mô hình xếp mẫu vào lớp ác tính theo ngưỡng đã chọn. Cần được chuyên gia y tế đánh giá.</div></div>',unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box result-benign"><div class="result-title">BENIGN — LÀNH TÍNH</div><div class="result-subtitle">Mô hình xếp mẫu vào lớp lành tính theo ngưỡng đã chọn. Kết quả không loại trừ hoàn toàn nguy cơ bệnh lý.</div></div>',unsafe_allow_html=True)
        x,y = st.columns(2)
        with x:
            st.metric("Xác suất Malignant (ác tính)",f"{p_m*100:.2f}%")
            st.progress(float(np.clip(p_m,0,1)))
        with y:
            st.metric("Xác suất Benign (lành tính)",f"{p_b*100:.2f}%")
            st.progress(float(np.clip(p_b,0,1)))
        chart = pd.DataFrame({"Xác suất (%)":[p_m*100,p_b*100]},index=["Malignant","Benign"])
        st.markdown("#### Biểu đồ xác suất")
        st.bar_chart(chart)
        st.caption(f"Ngưỡng đang dùng: {threshold:.5f}. Xác suất mô hình không đồng nghĩa với xác suất chẩn đoán lâm sàng.")

elif page == "Đánh giá mô hình":
    st.markdown("## Đánh giá mô hình")
    st.caption("Kiểm tra dự đoán trên bộ dữ liệu Breast Cancer Wisconsin gồm 569 mẫu.")
    st.warning("Lưu ý: các mẫu này đã được dùng khi huấn luyện mô hình cuối; chỉ số này không phải kết quả kiểm định độc lập.")
    if st.button("KIỂM TRA TOÀN BỘ 569 MẪU", use_container_width=True):
        X_all = df[features]
        y_true = df["diagnosis"].to_numpy(dtype=int)
        probs = model.predict_proba(X_all)
        malignant_idx = list(model.classes_).index(1)
        p_m = probs[:,malignant_idx]
        preds = (p_m >= threshold).astype(int)
        total = len(y_true)
        correct = int((preds==y_true).sum())
        wrong = total-correct
        accuracy = correct/total
        a,b,c,d = st.columns(4)
        a.metric("Tổng số mẫu",total)
        b.metric("Dự đoán đúng",correct)
        c.metric("Dự đoán sai",wrong)
        d.metric("Accuracy",f"{accuracy*100:.2f}%")
        st.progress(float(np.clip(accuracy,0,1)))
        tp = int(((y_true==1)&(preds==1)).sum())
        fn = int(((y_true==1)&(preds==0)).sum())
        tn = int(((y_true==0)&(preds==0)).sum())
        fp = int(((y_true==0)&(preds==1)).sum())
        st.markdown("### Ma trận nhầm lẫn")
        cm = pd.DataFrame([[tn,fp],[fn,tp]],index=["Thực tế Benign","Thực tế Malignant"],columns=["Dự đoán Benign","Dự đoán Malignant"])
        st.dataframe(cm,use_container_width=True)
        st.markdown("### Các mẫu dự đoán sai")
        wrong_idx = np.where(preds!=y_true)[0]
        if len(wrong_idx)==0:
            st.success("Không phát hiện mẫu dự đoán sai.")
        else:
            records=[]
            for i in wrong_idx:
                records.append({
                    "Mẫu số":int(i+1),
                    "Nhãn thực tế":"Malignant (ác tính)" if y_true[i]==1 else "Benign (lành tính)",
                    "Mô hình dự đoán":"Malignant (ác tính)" if preds[i]==1 else "Benign (lành tính)",
                    "Xác suất Malignant (%)":round(float(p_m[i])*100,2)
                })
            wrong_df=pd.DataFrame(records)
            st.write(f"Phát hiện **{len(wrong_df)} mẫu dự đoán sai**.")
            st.dataframe(wrong_df,use_container_width=True,hide_index=True)
            st.download_button("Tải danh sách mẫu dự đoán sai (CSV)",wrong_df.to_csv(index=False).encode("utf-8-sig"),"cac_mau_du_doan_sai.csv","text/csv",use_container_width=True)

st.markdown("""
<div class="footer"><b>BREASTCARE AI</b><br>
Hệ thống hỗ trợ phân loại khối u vú bằng máy học<br>
Breast Cancer Wisconsin Dataset · SVM RBF<br>
Công cụ học tập và nghiên cứu, không thay thế chẩn đoán y khoa.</div>
""",unsafe_allow_html=True)
