# The task is to identify the fever by naming it and establishing a threshold of 37.5°C or lower, containing of 
# Define Classification Variables
# Train Logistic Regression and Evaluation
# Train Neural Network Classification and Evaluation
# Train XGBoost Classification and Evaluation
# Compare model performance

# Extract Target Variables (with Labels)

y_class_F = (df_encoded["aveOralF"] >= 37.5).astype(int)  # Fever based on Fast Mode
y_class_M = (df_encoded["aveOralM"] >= 37.5).astype(int)  # Fever based on Monitor Mode

# Split the data for classification
# Assuming your selected features are in X_filtered
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

# Split data for classification task
X_train_class_F, X_test_class_F, y_train_class_F, y_test_class_F = train_test_split(
    X_scaled, y_class_F, test_size=0.2, random_state=42, stratify=y_class_F
)

X_train_class_M, X_test_class_M, y_train_class_M, y_test_class_M = train_test_split(
    X_scaled, y_class_M, test_size=0.2, random_state=42, stratify=y_class_M
)

# Check shape
print("Training set shape for Fever_F:", X_train_class_F.shape, y_train_class_F.shape)
print("Training set shape for Fever_M:", X_train_class_M.shape, y_train_class_M.shape)

# Initialize SMOTE
smote = SMOTE(sampling_strategy=0.7, random_state=42)

# Apply SMOTE to training data
X_train_balanced_F, y_train_balanced_F = smote.fit_resample(X_train_class_F, y_train_class_F)
X_train_balanced_M, y_train_balanced_M = smote.fit_resample(X_train_class_M, y_train_class_M)

# Check new class distribution
unique_F_bal, counts_F_bal = np.unique(y_train_balanced_F, return_counts=True)
unique_M_bal, counts_M_bal = np.unique(y_train_balanced_M, return_counts=True)

print("Balanced Class Distribution for Fever_F:", dict(zip(unique_F_bal, counts_F_bal)))
print("Balanced Class Distribution for Fever_M:", dict(zip(unique_M_bal, counts_M_bal)))

# Logistic Regression and Evaluation
# Train Logistic Regression Model
# Initialize Logistic Regression
log_reg_F = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
log_reg_M = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)

# Train the models
log_reg_F.fit(X_train_balanced_F, y_train_balanced_F)
log_reg_M.fit(X_train_balanced_M, y_train_balanced_M)

# Make predictions
y_pred_log_F = log_reg_F.predict(X_test_selected_scaled)
y_pred_log_M = log_reg_M.predict(X_test_selected_scaled)

# Evaluate performance
print("Logistic Regression Performance for Fever_F:")
print(classification_report(y_test_class_F, y_pred_log_F))
print(f"Accuracy: {accuracy_score(y_test_class_F, y_pred_log_F):.3f}")

print("\nLogistic Regression Performance for Fever_M:")
print(classification_report(y_test_class_M, y_pred_log_M))
print(f"Accuracy: {accuracy_score(y_test_class_M, y_pred_log_M):.3f}")

# Get probabilities instead of hard labels
y_prob_log_F = log_reg_F.predict_proba(X_test_class_F)[:, 1]  # Prob for class 1
y_prob_log_M = log_reg_M.predict_proba(X_test_class_M)[:, 1]  

# Adjust threshold
threshold = 0.5  # Lower threshold to improve recall
y_pred_log_F = (y_prob_log_F >= threshold).astype(int)
y_pred_log_M = (y_prob_log_M >= threshold).astype(int)

# Evaluate again
print("\nLogistic Regression Performance for Fever_F (Threshold 0.5):")
print(classification_report(y_test_class_F, y_pred_log_F))
print(f"Accuracy: {accuracy_score(y_test_class_F, y_pred_log_F):.3f}")

print("\nLogistic Regression Performance for Fever_M (Threshold 0.5):")
print(classification_report(y_test_class_M, y_pred_log_M))
print(f"Accuracy: {accuracy_score(y_test_class_M, y_pred_log_M):.3f}")

# Neural Network Classification and Evaluation
# Define Neural Network Model
nn_clf_F = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_class_F.shape[1],)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.1),
    Dense(1, activation="sigmoid")  # Sigmoid for binary classification
])

nn_clf_M = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_class_M.shape[1],)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.1),
    Dense(1, activation="sigmoid")
])

# Compile Model
nn_clf_F.compile(optimizer=Adam(learning_rate=0.01), loss="binary_crossentropy", metrics=["accuracy"])
nn_clf_M.compile(optimizer=Adam(learning_rate=0.01), loss="binary_crossentropy", metrics=["accuracy"])

# Use Early Stopping to prevent overfitting
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# Train Model
nn_clf_F.fit(X_train_class_F, y_train_class_F, validation_data=(X_test_class_F, y_test_class_F),
             epochs=300, batch_size=16, verbose=1, callbacks=[early_stop])

nn_clf_M.fit(X_train_class_M, y_train_class_M, validation_data=(X_test_class_M, y_test_class_M),
             epochs=300, batch_size=16, verbose=1, callbacks=[early_stop])

# Make Predictions
y_pred_nn_F = (nn_clf_F.predict(X_test_class_F) >= 0.5).astype(int)
y_pred_nn_M = (nn_clf_M.predict(X_test_class_M) >= 0.5).astype(int)

# Evaluate Performance
print("\nNeural Network Classification Performance for Fever_F:")
print(classification_report(y_test_class_F, y_pred_nn_F))
print(f"Accuracy: {accuracy_score(y_test_class_F, y_pred_nn_F):.3f}")

print("\nNeural Network Classification Performance for Fever_M:")
print(classification_report(y_test_class_M, y_pred_nn_M))
print(f"Accuracy: {accuracy_score(y_test_class_M, y_pred_nn_M):.3f}")

# XGBoost Classification and Evaluation
# Compute class weights (handle imbalance)
sample_weights_F = compute_sample_weight(class_weight="balanced", y=y_train_balanced_F)
sample_weights_M = compute_sample_weight(class_weight="balanced", y=y_train_balanced_M)

# Define XGBoost Classifier with optimized hyperparameters
xgb_clf_F = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
xgb_clf_M = xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)

# Train on balanced dataset with class weights
xgb_clf_F.fit(X_train_balanced_F, y_train_balanced_F, sample_weight=sample_weights_F)
xgb_clf_M.fit(X_train_balanced_M, y_train_balanced_M, sample_weight=sample_weights_M)

# Make Predictions
y_pred_xgb_F = xgb_clf_F.predict(X_test_class_F)
y_pred_xgb_M = xgb_clf_M.predict(X_test_class_M)

# Evaluate Performance
print("\nXGBoost Classification Performance for Fever_F:")
print(classification_report(y_test_class_F, y_pred_xgb_F))
print(f"Accuracy: {accuracy_score(y_test_class_F, y_pred_xgb_F):.3f}")

print("\nXGBoost Classification Performance for Fever_M:")
print(classification_report(y_test_class_M, y_pred_xgb_M))
print(f"Accuracy: {accuracy_score(y_test_class_M, y_pred_xgb_M):.3f}")

# Compare model performance
# Store results in a DataFrame
classification_results = pd.DataFrame({
    "Model": ["Logistic Regression", "Neural Network", "XGBoost"],
    "Accuracy (Fever_F)": [0.936, 0.961, 0.946],
    "Precision (Fever_F)": [0.50, 0.69, 0.55],
    "Recall (Fever_F)": [0.92, 0.69, 0.85],
    "F1-Score (Fever_F)": [0.65, 0.69, 0.67],
    "Accuracy (Fever_M)": [0.926, 0.966, 0.936],
    "Precision (Fever_M)": [0.60, 0.94, 0.66],
    "Recall (Fever_M)": [0.95, 0.73, 0.86],
    "F1-Score (Fever_M)": [0.74, 0.82, 0.75]
})

# Display comparison
print("\nClassification Model Performance Comparison")
print(classification_results)

# If you want to compare and evaluate binary classification thresholds
# You can do Threshold Tuning with ROC & PR Curves
