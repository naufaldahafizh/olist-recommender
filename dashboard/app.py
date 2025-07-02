import streamlit as st
import pandas as pd
import pickle
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load model
with open("./dashboard/cb_model.pkl", "rb") as f:
    cb_model = pickle.load(f)
with open("./dashboard/cf_model.pkl", "rb") as f:
    cf_model = pickle.load(f)

# Load data produk & customer
products = pd.read_csv("./data/olist_products_dataset.csv")
customers = pd.read_csv("./data/olist_customers_dataset.csv")
product_ids = products['product_id'].unique()
customer_ids = customers['customer_id'].unique()

st.title("Recommender System App")

menu = st.sidebar.selectbox("Pilih Jenis Rekomendasi", ["Content-Based", "Collaborative Filtering"])

if menu == "Content-Based":
    selected_product = st.selectbox("Pilih Produk", product_ids)
    n = st.slider("Jumlah Rekomendasi", 1, 10, 5)

    results = cb_model.recommend(selected_product, n=n)
    st.subheader("Rekomendasi Produk Serupa")
    st.dataframe(results)

elif menu == "Collaborative Filtering":
    selected_customer = st.selectbox("Pilih Customer", customer_ids)
    if st.button("Tampilkan Rekomendasi"):
        try:
            recommendations = cf_model.get_recommendations(selected_customer, top_n=5)
            st.subheader("Top 5 Rekomendasi:")
            for pid, score in recommendations:
                st.write(f"Produk: `{pid}` — Estimasi Rating: `{score:.2f}`")
        except Exception as e:
            st.error(f"Error: {e}")
            

