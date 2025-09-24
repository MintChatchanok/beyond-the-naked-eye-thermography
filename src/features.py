# 2.2.1 Missing value removal
print(df.isnull().sum())

df.dropna(inplace=True)

# 2.2.2 Categorical Encoding
# Identify categorical columns
categorical_cols = ["Gender", "Age", "Ethnicity"]

# Apply One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Check updated dataset
print(df_encoded.info())
print(df_encoded.head())

# 2.2.3 Feature Selection
# a.Correlation Matrix
# Compute correlation matrix
corr_matrix = df_encoded.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()

# b.XGBoost Feature Importance
# Define features and target variable
X = df_encoded.drop(columns=["aveOralF", "aveOralM"])  # Drop target columns
y = df_encoded["aveOralF"]  # Example: Predicting oral temperature

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
xgb_reg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_reg.fit(X_train, y_train)

# Get feature importance
importance_values = xgb_reg.feature_importances_
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance_values})
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

# Plot feature importance
xgb.plot_importance(xgb_reg, importance_type="weight", title="Feature Importance for aveOralF")
plt.show()

# Display top 20 features
print("Top 20 Features:\n", feature_importance.head(20))

# c.Drop Low-Importance Features
# Drop features with importance < 0.01
low_importance_features = feature_importance[feature_importance["Importance"] < 0.01]["Feature"].tolist()
X_filtered = X.drop(columns=low_importance_features)

print(f"Removed {len(low_importance_features)} low-importance features.")
print(f"Remaining features: {X_filtered.shape[1]}")

# d.Check Feature Importance & Recursive Feature Elimination (RFE)
#Check Feature Importance
X_filtered = X.drop(columns=low_importance_features)

# Train-test split
X_train_filtered, X_test_filtered, y_train_reg_F, y_test_reg_F = train_test_split(
    X_filtered, y, test_size=0.2, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train_filtered_scaled = scaler.fit_transform(X_train_filtered)
X_test_filtered_scaled = scaler.transform(X_test_filtered)

# XGBoost Feature Importance

xgb_model = XGBRegressor()
xgb_model.fit(X_train_filtered_scaled, y_train_reg_F)

feature_importance = xgb_model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]
important_features = X_filtered.columns[sorted_idx]

print("Top Important Features:")
print(important_features[:15])
# Split filtered data
X_train_filtered, X_test_filtered, y_train_filtered, y_test_filtered = train_test_split(
    X_filtered, y, test_size=0.2, random_state=42
)

# Use Linear Regression for RFE
estimator = LinearRegression()
selector = RFE(estimator, n_features_to_select=15)

# Fit RFE model
selector.fit(X_train_filtered, y_train_filtered)

# Get selected features
selected_features = X_filtered.columns[selector.support_]
print("Selected Features:", selected_features)

# Apply selected features to both train and test
X_train_selected = X_train_filtered[selected_features]
X_test_selected = X_test_filtered[selected_features]

# Debugging: Check feature counts
print(f"Training Features Count: {X_train_selected.shape[1]}")
print(f"Test Features Count: {X_test_selected.shape[1]}")

# 2.2.4 Polynomial Feature Expansion
degree = 2
poly = PolynomialFeatures(degree=degree, include_bias=False)

# Apply polynomial expansion
X_train_poly = poly.fit_transform(X_train_selected)
X_test_poly = poly.transform(X_test_selected)

# Check feature expansion
print(f"Original Feature Count: {X_train_selected.shape[1]}")
print(f"Polynomial Feature Count: {X_train_poly.shape[1]}")

# 2.2.5 Data Normalization
# Fit StandardScaler on training data
scaler = StandardScaler()
X_train_selected_scaled = scaler.fit_transform(X_train_selected)
X_test_selected_scaled = scaler.transform(X_test_selected)

# Debugging: Check feature count again
print(f"Train Features After Scaling: {X_train_selected_scaled.shape[1]}")
print(f"Test Features After Scaling: {X_test_selected_scaled.shape[1]}")

# Apply Feature Selection (Keep only selected features)
X_selected = X_filtered[selected_features]

# Apply StandardScaler to normalize the original features
scaler_features = StandardScaler()
X_selected_scaled = scaler_features.fit_transform(X_selected)

# Apply Polynomial Feature Expansion
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_selected_scaled)

# Apply StandardScaler to normalize the polynomial features
scaler_poly = StandardScaler()
X_poly_scaled = scaler_poly.fit_transform(X_poly)

# Convert back to DataFrame for readability
poly_feature_names = poly.get_feature_names_out(selected_features)
X_poly_scaled_df = pd.DataFrame(X_poly_scaled, columns=poly_feature_names)

# Check scaling
print(f"Mean after poly scaling (should be ~0): {X_poly_scaled.mean(axis=0)[:10]}")
print(f"Std deviation after poly scaling (should be ~1): {X_poly_scaled.std(axis=0)[:10]}")

# Verify Scaling
print("First few rows of original data:\n", X_filtered.head())
print("\nFirst few rows after scaling:\n", X_train_selected_scaled[:5])

