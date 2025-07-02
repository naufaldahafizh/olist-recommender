# Recommender System for E-Commerce (Olist Dataset)

This project builds a hybrid recommender system for an e-commerce platform using the Olist dataset. It includes content-based filtering, collaborative filtering, and interactive visualization using Streamlit.

---

## Project Structure

- `data/`: All Olist datasets (customers, orders, products, reviews, etc.)
- `notebooks/`: content based and collaborative filtering notebooks
- `src/`: Python modules for content-based and collaborative filtering
- `dashboard/`: Streamlit app and saved models
- `README.md`: Project documentation
- `requirements.txt`: All required Python packages

---

## Features

- **Content-Based Filtering**: Recommends similar products based on product category name.
- **Collaborative Filtering (SVD)**: Suggests products to customers based on ratings.
- **Hybrid Exploration**: Users can interactively try both methods via Streamlit app.

---

## Streamlit App Demo

```bash
cd app
streamlit run app.py
```

You'll be able to:
- Select a product → get similar product recommendations
- Select a customer → get top product recommendations based on behavior

## Modeling Techniques
- TF-IDF + Cosine Similarity (Content-Based)
- Matrix Factorization with SVD (Collaborative Filtering)

## Installation
`pip install -r requirements.txt`
