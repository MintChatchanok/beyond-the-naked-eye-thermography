# The task is to find the most effective model
# Linear Regression and Evaluation
# Polynomial Regression and Evaluation
# Neural Network and Evaluation
# XGBoost and Evaluation
# Then, Compare Model Performance: RMSE, R²

# Define Regression Targets
y_reg_F = df_encoded["aveOralF"]  # Regression Target: Oral Temperature (Fast Mode)
y_reg_M = df_encoded["aveOralM"]  # Regression Target: Oral Temperature (Monitor Mode)

# Regression Splits
X_train_reg_F, X_test_reg_F, y_train_reg_F, y_test_reg_F = train_test_split(X, y_reg_F, test_size=0.2, random_state=42)
X_train_reg_M, X_test_reg_M, y_train_reg_M, y_test_reg_M = train_test_split(X, y_reg_M, test_size=0.2, random_state=42)

#Train Linear Regression for aveOralF
lin_reg_F = LinearRegression()
lin_reg_F.fit(X_train_filtered_scaled, y_train_reg_F)

#Train Linear Regression for aveOralM
lin_reg_M = LinearRegression()
lin_reg_M.fit(X_train_filtered_scaled, y_train_reg_M)

#Make predictions on test set
y_pred_lin_F = lin_reg_F.predict(X_test_filtered_scaled)
y_pred_lin_M = lin_reg_M.predict(X_test_filtered_scaled)

#Compute RMSE & R² Score for aveOralF
rmse_lin_F = root_mean_squared_error(y_test_reg_F, y_pred_lin_F)
r2_lin_F = r2_score(y_test_reg_F, y_pred_lin_F)

#Compute RMSE & R² Score for aveOralM
rmse_lin_M = root_mean_squared_error(y_test_reg_M, y_pred_lin_M)
r2_lin_M = r2_score(y_test_reg_M, y_pred_lin_M)

#Print results
print("Linear Regression Performance")
print(f"RMSE (aveOralF): {rmse_lin_F:.3f}, R²: {r2_lin_F:.3f}")
print(f"RMSE (aveOralM): {rmse_lin_M:.3f}, R²: {r2_lin_M:.3f}")

# Polynomial Regression and Evaluation
# Split data into train and test
X_train_filtered, X_test_filtered, y_train, y_test = train_test_split(
    X_filtered, y, test_size=0.2, random_state=42
)

#Apply Feature Selection
X_train_selected = X_train_filtered[selected_features]
X_test_selected = X_test_filtered[selected_features]

#Apply StandardScaler to the selected features
scaler = StandardScaler()
X_train_selected_scaled = scaler.fit_transform(X_train_selected)
X_test_selected_scaled = scaler.transform(X_test_selected)

#Apply Polynomial Feature Expansion
degree = 2
poly = PolynomialFeatures(degree=degree, include_bias=False)

X_train_poly = poly.fit_transform(X_train_selected_scaled)
X_test_poly = poly.transform(X_test_selected_scaled)

#Print feature counts
print("Original Feature Count:", X_train_selected.shape[1])
print("Polynomial Feature Count:", X_train_poly.shape[1])

#Train Polynomial Regression Model
poly_reg_F = LinearRegression()
poly_reg_F.fit(X_train_poly, y_train)

poly_reg_M = LinearRegression()
poly_reg_M.fit(X_train_poly, y_train)

#Predict on test set
y_pred_poly_F = poly_reg_F.predict(X_test_poly)
y_pred_poly_M = poly_reg_M.predict(X_test_poly)

#Compute RMSE & R²
rmse_poly_F = root_mean_squared_error(y_test, y_pred_poly_F)
r2_poly_F = r2_score(y_test, y_pred_poly_F)

rmse_poly_M = root_mean_squared_error(y_test, y_pred_poly_M)
r2_poly_M = r2_score(y_test, y_pred_poly_M)

print("Polynomial Regression Performance")
print(f"RMSE (aveOralF): {rmse_poly_F:.3f}, R²: {r2_poly_F:.3f}")
print(f"RMSE (aveOralM): {rmse_poly_M:.3f}, R²: {r2_poly_M:.3f}")

# You optimise if you see some necessaries with ElasticNet & Hyperparameter Tuning
# But I will skip that part because it depends on the data

# Define Neural Network Model
nn_model_F = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_selected_scaled.shape[1],)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.1),
    Dense(1)  # Output layer for regression
])

nn_model_M = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_selected_scaled.shape[1],)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.1),
    Dense(1)  # Output layer for regression
])

# Compile Models
nn_model_F.compile(optimizer=Adam(learning_rate=0.01), loss="mse", metrics=["mae"])
nn_model_M.compile(optimizer=Adam(learning_rate=0.01), loss="mse", metrics=["mae"])

# Use Early Stopping
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# Train Model
nn_model_F.fit(X_train_selected_scaled, y_train_reg_F, validation_data=(X_test_selected_scaled, y_test_reg_F),
               epochs=300, batch_size=16, verbose=1, callbacks=[early_stop])

nn_model_M.fit(X_train_selected_scaled, y_train_reg_M, validation_data=(X_test_selected_scaled, y_test_reg_M),
               epochs=300, batch_size=16, verbose=1, callbacks=[early_stop])

# Make Predictions AFTER Training
y_pred_nn_F = nn_model_F.predict(X_test_selected_scaled).flatten()
y_pred_nn_M = nn_model_M.predict(X_test_selected_scaled).flatten()

# Compute RMSE
rmse_nn_F = np.sqrt(mean_squared_error(y_test_reg_F, y_pred_nn_F))
rmse_nn_M = np.sqrt(mean_squared_error(y_test_reg_M, y_pred_nn_M))

# Compute R² Score
r2_nn_F = r2_score(y_test_reg_F, y_pred_nn_F)
r2_nn_M = r2_score(y_test_reg_M, y_pred_nn_M)

# Print Results
print(f"Neural Network Regression - RMSE (aveOralF): {rmse_nn_F:.3f}, R²: {r2_nn_F:.3f}")
print(f"Neural Network Regression - RMSE (aveOralM): {rmse_nn_M:.3f}, R²: {r2_nn_M:.3f}")

# Define XGBoost Parameters
xgb_params = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5,
    "random_state": 42
}

# Train XGBoost Regression Model for aveOralF
xgb_reg_F = xgb.XGBRegressor(**xgb_params)
xgb_reg_F.fit(X_train_selected_scaled, y_train_reg_F)

# Train XGBoost Regression Model for aveOralM
xgb_reg_M = xgb.XGBRegressor(**xgb_params)
xgb_reg_M.fit(X_train_selected_scaled, y_train_reg_M)

# Make Predictions
y_pred_xgb_F = xgb_reg_F.predict(X_test_selected_scaled)
y_pred_xgb_M = xgb_reg_M.predict(X_test_selected_scaled)

# Evaluate Performance
rmse_xgb_F = root_mean_squared_error(y_test_reg_F, y_pred_xgb_F)
r2_xgb_F = r2_score(y_test_reg_F, y_pred_xgb_F)

rmse_xgb_M = root_mean_squared_error(y_test_reg_M, y_pred_xgb_M)
r2_xgb_M = r2_score(y_test_reg_M, y_pred_xgb_M)

# Print Results
print(f"XGBoost Regression - RMSE (aveOralF): {rmse_xgb_F:.3f}, R²: {r2_xgb_F:.3f}")
print(f"XGBoost Regression - RMSE (aveOralM): {rmse_xgb_M:.3f}, R²: {r2_xgb_M:.3f}")

# Store results in a DataFrame
model_comparison = pd.DataFrame({
    "Model": ["Linear Regression", 
              "Polynomial Regression", 
              "Polynomial + ElasticNet", 
              "Neural Network",
              "XGBoost"],
    
    "RMSE (aveOralF)": [rmse_lin_F, 
                         rmse_poly_F, 
                         rmse_elastic_F,
                         rmse_nn_F,
                         rmse_xgb_F],

    "R² (aveOralF)": [r2_lin_F, 
                       r2_poly_F,
                       r2_elastic_F,
                       r2_nn_F,
                       r2_xgb_F],

    "RMSE (aveOralM)": [rmse_lin_M, 
                         rmse_poly_M, 
                         rmse_elastic_M, 
                         rmse_nn_M,
                         rmse_xgb_M],

    "R² (aveOralM)": [r2_lin_M, 
                       r2_poly_M, 
                       r2_elastic_M,
                       r2_nn_M,
                       r2_xgb_M]
})

# Sort by best R² score
model_comparison = model_comparison.sort_values(by=["R² (aveOralF)", "R² (aveOralM)"], ascending=False)

# Display results
print("\nRegression Model Comparison:")
print(model_comparison)

# If you see that the results are not good enough
# You can Fine-Tuning XGBoost Regression with GridSearchCV
# Then, compare the results again
