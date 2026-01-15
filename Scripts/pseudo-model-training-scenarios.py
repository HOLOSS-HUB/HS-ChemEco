import data_utils
import model_utils
import evaluation_utils
import preprocessing
import persistence

# Load and preprocess data
data = data_utils.load_data("path_to_dataset")
features = data_utils.get_feature_columns()
targets = data_utils.get_target_columns()
categoricals = data_utils.get_categorical_columns()

data = preprocessing.encode_categorical(data, columns=categoricals)

# Transform targets
target_transformer = preprocessing.get_transformer()
Y = target_transformer.fit_transform(data[targets])
persistence.save_object(target_transformer, "transformer_path")

# Scale features
feature_matrix = data[features]
scaler = preprocessing.get_scaler()
X = scaler.fit_transform(feature_matrix)
persistence.save_object(scaler, "scaler_path")
data['is_synthetic'] = data.get('is_synthetic', default=0)
real_idx = data_utils.get_indices(data, condition='is_real')
synthetic_idx = data_utils.get_indices(data, condition='is_synthetic')

# Prepare training configuration
repeats, rounds = N, M
metrics_log = []
feature_importances = []

progress = initialize_progress_bar(total=repeats * rounds * len(targets))

# === (AUGMENTED + REINFORCED) TRAINING ===
for repeat in range(repeats):
    train_real, test_real = data_utils.split_indices(real_idx, ratio=0.8, seed=repeat)
    train_combined = data_utils.combine_indices(train_real, synthetic_idx)

    X_train, Y_train = X[train_combined], Y[train_combined]
    X_test, Y_test = X[test_real], Y[test_real]
    weights = data_utils.compute_weights(data, train_combined)

    for round in range(rounds):
        X_train_chain, X_test_chain = X_train.copy(), X_test.copy()
        predictions_train, predictions_test = [], []

        for i in range(len(targets)): #Example for the Optmized RF algorithm
            model = model_utils.build_model(
                bootstrap=True,
                n_estimators=450,
                max_features=350,
                max_depth=150)
            y_train, y_test = Y_train[:, i], Y_test[:, i]

            model.fit(X_train_chain, y_train, sample_weight=weights)
            y_pred_train = model.predict(X_train_chain)
            y_pred_test = model.predict(X_test_chain)

            predictions_train.append(y_pred_train)
            predictions_test.append(y_pred_test)

            if i == 0:
                feature_importances.append(model.feature_importances_)

            metrics_log.append({
                "repeat": repeat + 1,
                "round": round + 1,
                "target": targets[i],
                **evaluation_utils.compute_metrics(y_train, y_pred_train, y_test, y_pred_test)
            })

            # Chaining predictions as new input features
            X_train_chain = data_utils.add_prediction_column(X_train_chain, y_pred_train)
            X_test_chain = data_utils.add_prediction_column(X_test_chain, y_pred_test)

            progress.update(1)

        Y_pred_train_all = data_utils.stack_predictions(predictions_train)
        Y_pred_test_all = data_utils.stack_predictions(predictions_test)

        if repeat == 0 and round == 0:
            Y_test_reinforced = Y_test.copy()
            Y_pred_reinforced = Y_pred_test_all.copy()

        overall_metrics = evaluation_utils.compute_metrics(Y_train, Y_pred_train_all, Y_test, Y_pred_test_all)
        metrics_log.append({**overall_metrics, "repeat": repeat + 1, "round": round + 1})

progress.close()

# === AUGMENTED DATA ONLY ===
multi_metrics = []
multi_importances = []

for repeat in range(repeats):
    train_real, test_real = data_utils.split_indices(real_idx, ratio=0.8, seed=repeat)
    train_combined = data_utils.combine_indices(train_real, synthetic_idx)

    X_train, Y_train = X[train_combined], Y[train_combined]
    X_test, Y_test = X[test_real], Y[test_real]
    weights = data_utils.compute_weights(data, train_combined)

    model = model_utils.build_multi_output_model()
    model.fit(X_train, Y_train, sample_weight=weights)

    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)

    metrics = evaluation_utils.compute_metrics(Y_train, Y_pred_train, Y_test, Y_pred_test)
    multi_metrics.append({**metrics, "repeat": repeat + 1})

    importances = model_utils.average_feature_importance(model)
    multi_importances.append(importances)

    if repeat == 0:
        Y_test_augmented = Y_test.copy()
        Y_pred_augmented = Y_pred_test.copy()

# === REAL-ONLY TRAINING ===
realonly_metrics = []
realonly_importances = []

for repeat in range(repeats):
    train_real, test_real = data_utils.split_indices(real_idx, ratio=0.8, seed=repeat)

    X_train, Y_train = X[train_real], Y[train_real]
    X_test, Y_test = X[test_real], Y[test_real]

    model = model_utils.build_multi_output_model()
    model.fit(X_train, Y_train)

    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)

    metrics = evaluation_utils.compute_metrics(Y_train, Y_pred_train, Y_test, Y_pred_test)
    realonly_metrics.append({**metrics, "repeat": repeat + 1})

    importances = model_utils.average_feature_importance(model)
    realonly_importances.append(importances)

    if repeat == 0:
        Y_test_realonly = Y_test.copy()
        Y_pred_realonly = Y_pred_test.copy()

# Save metrics
persistence.save_metrics(metrics_log, "path_full_reinforced.csv")
persistence.save_metrics(multi_metrics, "path_augmented.csv")
persistence.save_metrics(realonly_metrics, "path_realonly.csv")

# Save importance tables
persistence.save_importances(feature_importances, "importances_full.csv")
persistence.save_importances(multi_importances, "importances_augmented.csv")
persistence.save_importances(realonly_importances, "importances_realonly.csv")

print("Models trained and evaluated.")
