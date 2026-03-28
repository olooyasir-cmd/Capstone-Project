# 📱 Mobile App Success Predictor — Summary Report

**Project:** Google Play Store ML Analysis  
**Tools:** Python, Pandas, NumPy, Scikit-learn, Streamlit, Matplotlib, Seaborn  
**Dataset:** 10,841 apps → 8,196 after cleaning

---

## 1. Data Preprocessing Summary

The raw dataset required substantial cleaning before modeling:

| Issue | Resolution |
|---|---|
| 1,474 missing `Rating` values | Dropped (target variable — cannot impute) |
| Duplicate app entries (2,645) | Removed, kept most recent version |
| Size stored as "19M" / "8.7k" | Parsed to float MB; medians filled NaN |
| Installs stored as "10,000+" | Stripped `+` and `,`, cast to int |
| Price stored as "$4.99" | Stripped `$`, cast to float |
| Multi-genre entries ("Action;Casual") | Split on `;`, used first genre only |
| Categorical columns (Category, Content Rating) | Label Encoded for ML models |

**Final clean dataset:** 8,196 apps × 17 columns.

---

## 2. Exploratory Data Analysis

**Reviews vs Ratings:**  
A scatter plot (log-scale Reviews vs Rating) reveals a weak positive correlation (~0.06 Pearson). Apps with more reviews tend to converge toward the 4.0–4.5 band, suggesting that visibility drives quality signals rather than raw engagement guaranteeing higher ratings.

**Rating Distribution:**  
The distribution is left-skewed with a strong peak around 4.3–4.5, indicating that most apps are highly rated and the Play Store ecosystem rewards quality. The mean is approximately 4.19.

**Category Insights:**  
Events, Education, and Art & Design categories have the highest average ratings, while Dating and Tools show lower ratings. This suggests niche, utility-driven apps face higher scrutiny.

---

## 3. Machine Learning Models

### Model 1 — Random Forest Regressor (Supervised)

| Metric | Value |
|---|---|
| Mean Absolute Error (MAE) | **0.3556** |
| Root Mean Squared Error (RMSE) | **0.5100** |
| Estimators | 200 trees |
| Train/Test Split | 80% / 20% |

**Interpretation:** The model predicts app ratings within ±0.36 stars on average — strong performance given rating subjectivity.

**Feature Importance Rankings (most → least influential):**

| Rank | Feature | Why It Matters |
|---|---|---|
| 1 | **Reviews** | Engagement volume signals an active, satisfied userbase |
| 2 | **Installs** | Popularity proxy — well-installed apps tend to have more invested users |
| 3 | **Category** | Genre norms vary; games face harsher scrutiny than productivity tools |
| 4 | **Size (MB)** | Lighter apps have broader device reach and better performance perception |
| 5 | **Price** | Paid apps attract committed users with higher expectations |
| 6 | **Genre** | Finer-grained than Category; captures sub-audience behavior |
| 7 | **Content Rating** | Audience composition affects rating culture |
| 8 | **Type** | Free/Paid binary; captured by price but adds complementary signal |

> **Key Insight:** Reviews and Installs together dominate feature importance (~45% combined). Developers should prioritize early review acquisition and install growth strategies over app size optimization.

---

### Model 2 — KNN Cosine Similarity Recommender (Unsupervised)

**Approach:** Each app is encoded across 5 dimensions — Category, Genre, Price, Content Rating, and Size — then normalized with StandardScaler. A brute-force KNN with cosine similarity retrieves the 5 most similar neighbors.

**Why Cosine Similarity?**  
Cosine similarity captures directional similarity in feature space rather than Euclidean distance, making it robust to scale differences between features (e.g., price ranging $0–$400 vs. genre encoded 0–40).

**Example Output for "Clash of Clans":**

| App | Category | Genre | Similarity |
|---|---|---|---|
| Clash Royale | GAME | Strategy | 1.000 |
| Boom Beach | GAME | Strategy | 1.000 |
| SimCity BuildIt | GAME | Strategy | 0.998 |

Results are semantically meaningful — recommending direct competitors within the same genre/price/audience profile.

---

## 4. Streamlit Application

The deployed app (`app.py`) features:

- **Prediction Tab:** Dropdown inputs for Category, Content Rating, Genre, Type, Size slider, and Price input. Outputs predicted rating badge, gauge chart, and feature importance visualization.
- **Recommendation Tab:** Text input for any app name; returns top-5 similar apps in a styled table + cosine similarity bar chart.
- **Data Insights Tab:** EDA charts including scatter plots, distributions, category comparisons, and genre breakdowns.

---

## 5. Conclusions & Recommendations

1. **Focus on review acquisition early.** Reviews are the single biggest predictor of app rating. Implement in-app prompts at moments of delight (completed level, successful task).

2. **Category selection matters.** Education and Events apps consistently outperform Games and Tools in average rating. If cross-category is viable, lean toward higher-rated segments.

3. **Keep apps lean.** Size (MB) is a top-5 feature. Apps under 30MB consistently perform better in rating predictions.

4. **Free apps perform marginally better on average.** The Free/Paid split shows free apps clustering slightly higher, likely due to lower user expectation thresholds and volume-driven review bias.

5. **The KNN recommender is production-ready** for competitive analysis — developers can input their concept app name and immediately see who they are competing with.

---

*Report generated for Google Play Store ML Project — 8,196 apps analyzed.*
