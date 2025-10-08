import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data
def load_data():
    restaurants = pd.read_csv('data/zomato.csv', encoding='latin-1')
    restaurants = restaurants[['Restaurant Name','Cuisines','Aggregate rating','Average Cost for two','City']]
    restaurants = restaurants.dropna(subset=['Cuisines','Aggregate rating'])
    restaurants.reset_index(inplace=True)
    restaurants.rename(columns={'index':'restaurant_id'}, inplace=True)
    return restaurants

restaurants = load_data()

# Dummy user ratings
users = [101, 102, 103, 104, 105]
ratings = pd.DataFrame({
    'user_id': np.random.choice(users, 100),
    'restaurant_id': np.random.choice(restaurants['restaurant_id'], 100),
    'rating': np.random.randint(1, 6, 100)
})

# -----------------------------
# Content-Based Filtering
# -----------------------------
restaurants_encoded = restaurants[['Cuisines', 'Average Cost for two']].copy()
restaurants_encoded['Cuisines'] = restaurants_encoded['Cuisines'].apply(lambda x: x.split(',')[0])
restaurants_encoded = pd.get_dummies(restaurants_encoded, columns=['Cuisines'])

content_similarity = cosine_similarity(restaurants_encoded)
content_similarity_df = pd.DataFrame(content_similarity, index=restaurants['restaurant_id'], columns=restaurants['restaurant_id'])

def content_based_recommend(user_ratings, top_n=5):
    recommended = {}
    for rest_id in user_ratings['restaurant_id']:
        sims = content_similarity_df[rest_id].sort_values(ascending=False)
        for sim_rest, score in sims.items():
            if sim_rest not in user_ratings['restaurant_id'].values:
                recommended[sim_rest] = recommended.get(sim_rest, 0) + score
    recommended_sorted = sorted(recommended.items(), key=lambda x: x[1], reverse=True)
    top_restaurants = [rest_id for rest_id, _ in recommended_sorted[:top_n]]
    return restaurants[restaurants['restaurant_id'].isin(top_restaurants)][['restaurant_id', 'Restaurant Name']]

# -----------------------------
# Collaborative Filtering
# -----------------------------
user_item_matrix = ratings.pivot(index='user_id', columns='restaurant_id', values='rating').fillna(0)
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

def collaborative_filtering_recommend(user_id, top_n=5):
    if user_id not in user_item_matrix.index:
        return restaurants.sample(top_n)[['restaurant_id', 'Restaurant Name']]
    sim_scores = user_similarity_df[user_id]
    weighted_ratings = user_item_matrix.T.dot(sim_scores)
    already_rated = ratings[ratings['user_id'] == user_id]['restaurant_id'].values
    recs = [(rid, score) for rid, score in zip(user_item_matrix.columns, weighted_ratings) if rid not in already_rated]
    recs_sorted = sorted(recs, key=lambda x: x[1], reverse=True)[:top_n]
    top_restaurants = [rid for rid, _ in recs_sorted]
    return restaurants[restaurants['restaurant_id'].isin(top_restaurants)][['restaurant_id', 'Restaurant Name']]

# -----------------------------
# Hybrid Recommendation
# -----------------------------
def hybrid_recommend(user_id, top_n=5, alpha=0.5):
    user_ratings = ratings[ratings['user_id'] == user_id]
    cb_recs = content_based_recommend(user_ratings, top_n=10)
    cb_scores = {rid: 1 / (idx + 1) for idx, rid in enumerate(cb_recs['restaurant_id'])}
    cf_recs = collaborative_filtering_recommend(user_id, top_n=10)
    cf_scores = {rid: 1 / (idx + 1) for idx, rid in enumerate(cf_recs['restaurant_id'])}
    combined_scores = {}
    for rid in set(list(cb_scores.keys()) + list(cf_scores.keys())):
        combined_scores[rid] = alpha * cb_scores.get(rid, 0) + (1 - alpha) * cf_scores.get(rid, 0)
    top_recs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_restaurants = [rid for rid, _ in top_recs]
    return restaurants[restaurants['restaurant_id'].isin(top_restaurants)][['restaurant_id', 'Restaurant Name']]

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🍽️ Hybrid Restaurant Recommender System")

st.sidebar.header("User Input")
user_id = st.sidebar.selectbox("Select User ID", [101, 102, 103, 104, 105])
top_n = st.sidebar.slider("Number of Recommendations", 1, 10, 5)
alpha = st.sidebar.slider("Hybrid Weight (0=Collaborative, 1=Content)", 0.0, 1.0, 0.5)

if st.button("Get Recommendations"):
    recs = hybrid_recommend(user_id, top_n=top_n, alpha=alpha)
    st.success(f"Top {top_n} restaurant recommendations for User {user_id}:")
    st.dataframe(recs)

