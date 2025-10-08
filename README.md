# 🍽️ Hybrid Restaurant Recommender System

A **hybrid restaurant recommendation system** combining **content-based filtering** and **collaborative filtering** to suggest restaurants. This project includes a **Streamlit frontend** for easy interaction, allowing users to get personalized restaurant recommendations.

---

## 📝 Project Overview

The goal of this project is to build a **hybrid recommendation system** for restaurants using Python. It combines:

1. **Content-Based Filtering** – recommends restaurants similar to ones the user has rated highly, based on features like cuisine type and average cost.
2. **Collaborative Filtering** – recommends restaurants based on similar users’ ratings.
3. **Hybrid Approach** – combines both methods for better accuracy and personalization.

The system includes a **Streamlit web interface** where users can select their ID, number of recommendations, and hybrid weight to see top restaurant suggestions.

---

## 📊 Dataset

* **Source:** Zomato Restaurants Dataset
* **File:** `zomato.csv`
* **Columns used:**

  * `Restaurant Name`
  * `Cuisines`
  * `Aggregate rating`
  * `Average Cost for two`
  * `City`

> Note: User ratings are randomly generated for demonstration purposes.

---

## 🚀 Features

* **Hybrid Recommendation** combining content-based and collaborative filtering
* **Interactive Streamlit UI**:

  * Select User ID
  * Adjust number of recommendations
  * Adjust hybrid weight (content vs collaborative)
* **Instant display** of recommended restaurants

---

## 🧠 Tech Stack

* **Python**
* **Libraries:** Pandas, NumPy, Scikit-learn, Streamlit

---

## 📂 Folder Structure

```
hybrid_recommender/
├── data/
│   └── zomato.csv
├── hybrid_recommender.py
├── app.py
└── README.md
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/hybrid_recommender.git
   cd hybrid_recommender
   ```

2. **Install dependencies**

   ```bash
   pip install pandas numpy scikit-learn streamlit
   ```

3. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

4. **Interact with the app**

   * Select a **User ID**
   * Choose **Number of recommendations**
   * Adjust the **Hybrid weight** slider (0 = fully collaborative, 1 = fully content-based)
   * Click **Get Recommendations** to see the results

---

## 🖥️ How It Works

1. **Load Dataset:** Load the Zomato dataset and select relevant columns. Assign IDs to restaurants.
2. **Generate Ratings:** Random ratings are generated for demonstration.
3. **Content-Based Filtering:** Uses cosine similarity of one-hot encoded cuisines to recommend similar restaurants.
4. **Collaborative Filtering:** Uses user-item matrix and cosine similarity between users to recommend based on similar users.
5. **Hybrid Recommendation:** Combines content-based and collaborative scores using a weighting parameter (`alpha`).
6. **Streamlit UI:** Users can interactively get recommendations based on their inputs.

---

## 🧾 Output

The app displays the **top recommended restaurants** in a table. Example:

| restaurant_id | Restaurant Name  |
| ------------- | ---------------- |
| 45            | The Spice Lounge |
| 87            | Cafe Delight     |
| 12            | Gourmet Hub      |
| 102           | Bistro Central   |
| 9             | Curry House      |

---

## 👩‍💻 Author

**Ritushree Bohara**
💼 Software Developer | Fintech & Web Development Enthusiast
📧 [bohraritushree@gmail.com](mailto:bohraritushree@gmail.com)


---

## 📌 Notes

* Dataset: Zomato Restaurants dataset
* Ratings are **randomly generated** for demonstration
* Can be extended with **real user ratings** for production-ready use
* Hybrid weight (`alpha`) allows testing different combinations of content and collaborative filtering
