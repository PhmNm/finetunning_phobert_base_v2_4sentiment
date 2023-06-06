import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model import *

### Load data, model
df = pd.read_csv('data.csv')
count_df = df['label_word'].value_counts()

### Web content
# Title
st.title("MÔ HÌNH PHÂN LOẠI CẢM XÚC TIẾNG VIỆT")

# Dataset
st.subheader("Bộ Dữ Liệu")

st.write(df)

# Plot data
st.subheader("Chi Tiết Bộ Dữ Liệu")

fig, ax = plt.subplots()
ax = count_df.plot.bar(color=['lightgreen','brown','wheat'])
plt.gcf().autofmt_xdate()

for container in ax.containers:
    ax.bar_label(container)

st.pyplot(fig)

# Model info
st.subheader("Mô Hình")

st.write("Mô hình phân loại đã xây dựng sử dụng kiến trúc base_v2 của [PhoBERT](https://github.com/VinAIResearch/PhoBERT).")

# Prediction
st.subheader("Dự Đoán Cảm Xúc")

pred = -1
value = st.text_input("Nhập một câu tiếng Việt:")

if value != "":
    pred = predict({
        "inputs": value,
        "wait_for_model": True,
    })

if pred == 'LABEL_0':
    st.write("Câu này mang tính tiêu cực!")
elif pred == 'LABEL_1':
    st.write("Câu này mang tính trung tính!")
elif pred == 'LABEL_2':
    st.write("Câu này mang tính tích cực!")