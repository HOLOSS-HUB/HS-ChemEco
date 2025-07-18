import data_handler
import model_builder
import transformation_tools
import feature_selection_tools
import persistence_manager
import uncertainty_estimation

# Initialize storage and paths
storage_path = data_handler.initialize_storage(location="user_defined_path")
raw_data = data_handler.load_data(path="dataset_path.csv")
target_variables = data_handler.get_target_variables()
categorical_features = data_handler.get_categorical_variables()
numerical_features = data_handler.get_numerical_descriptors()

# Encode categorical features
processed_data = data_handler.encode_categorical(raw_data, columns=categorical_features)
processed_data = data_handler.drop_missing(processed_data, columns=target_variables)
selected_columns = data_handler.get_combined_features(numerical_features, processed_data, prefix=categorical_features)
X_raw = data_handler.clean_feature_matrix(processed_data, selected_columns)

# Feature selection
selector = feature_selection_tools.get_variance_selector(threshold=threshold_value)
X_selected = selector.fit_transform(X_raw)
selected_feature_names = selector.get_selected_feature_names()
persistence_manager.save(selected_feature_names, "feature_list_path")

# Transform targets
target_transformer = transformation_tools.get_quantile_transformer()
Y_transformed = target_transformer.fit_transform(processed_data[target_variables])
persistence_manager.save(target_transformer, "target_transformer_path")

# Scale features
feature_scaler = transformation_tools.get_scaler()
X_scaled = feature_scaler.fit_transform(X_selected)
persistence_manager.save(feature_scaler, "feature_scaler_path")

# Identify real data subset
real_data_indices = data_handler.get_indices_by_condition(processed_data, condition="is_real")

# Define model constructor
def build_ensemble(seed=None):
    return model_builder.get_ensemble_model(seed=seed)

# Train chained ensemble models
prediction_means = []
prediction_stds = []
X_chain_input = X_scaled.copy()
seeds = [list_of_random_seeds]
weights = data_handler.assign_sample_weights(processed_data)

for idx, target in enumerate(target_variables):
    target_column = Y_transformed[:, idx]
    ensemble_predictions = []

    for seed in seeds:
        model = build_ensemble(seed)
        model.fit(X_chain_input, target_column, sample_weight=weights)
        persistence_manager.save_model(model, f"model_target_{idx}_seed_{seed}")
        ensemble_predictions.append(model.predict(X_chain_input))

    mean_prediction = transformation_tools.aggregate_mean(ensemble_predictions)
    std_prediction = transformation_tools.aggregate_std(ensemble_predictions)

    prediction_means.append(mean_prediction)
    prediction_stds.append(std_prediction)

    X_chain_input = data_handler.append_column(X_chain_input, mean_prediction)

# Post-process predictions
predicted_outputs = transformation_tools.inverse_transform(target_transformer, prediction_means)
predicted_uncertainty = transformation_tools.inverse_transform(target_transformer, prediction_stds)

# Compute percentage-based uncertainty
percentage_errors = []

for idx, target in enumerate(target_variables):
    mu = predicted_outputs[:, idx]
    sigma = predicted_uncertainty[:, idx]
    percent_error = uncertainty_estimation.compute_percentage_error(mu, sigma)
    percentage_errors.append(percent_error)
    
for idx, target in enumerate(target_variables):
    model = model_builder.get_linear_regression()
    model.fit(predicted_outputs[:, idx].reshape(-1, 1), predicted_uncertainty[:, idx])
    persistence_manager.save_model(model, f"uncertainty_model_{target}")

error_table = data_handler.create_error_table(
    errors=percentage_errors,
    labels=[f"{var}_error" for var in target_variables],
    indices=real_data_indices
)
persistence_manager.save_table(error_table, "uncertainty_summary_path")

print("All ensemble models trained. Predictions processed and uncertainty estimators saved.")
