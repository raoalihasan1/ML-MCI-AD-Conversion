import numpy as np
import pandas as pd
import random
import seaborn as sns
from enum import Enum
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from itertools import combinations
from matplotlib import pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class Over_samplers(Enum):
    RANDOM = "Random"
    SMOTE = "SMOTE"
    ADASYN = "ADASYN"


def create_ensemble_model(
    cv_results: dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_train: pd.Series,
    Y_test: pd.Series,
) -> dict:
    """
    Create and evaluate ensemble models (StackingClassifier and VotingClassifier)
    using the best estimators and returns the best ensemble model.

    Args:
        dict: The results from the `tune_hyperparameters` function.
        pd.DataFrame: The training data features.
        pd.DataFrame: The testing data features.
        pd.Series: The training data labels.
        pd.Series: The testing data labels.

    Returns:
        dict: A dictionary containing the information of the best ensemble model:
            - "type" (str): ("StackingClassifier" or "VotingClassifier").
            - "model" (object): The best ensemble model.
            - "accuracy" (float): The accuracy of the best ensemble model.
            - "estimators" (list): The base estimators used in the ensemble.
            - "predicted_labels" (ndarray): Predicted labels for the test data.
            - "predicted_prob" (ndarray): Predicted probabilities for the test set.
            - "metrics" (dict): A dictionary of performance metrics.
    """
    model_accuracy = lambda prediction: accuracy_score(Y_test, prediction)
    predicted_prob = lambda model: (
        model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    )
    compute_metrics = lambda y_true, y_pred: {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Error Rate": 1 - accuracy_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred, average="weighted"),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "Mean Squared Error": mean_squared_error(y_true, y_pred),
        "Mean Absolute Error": mean_absolute_error(y_true, y_pred),
    }

    # Get the best estimators from the cv_results dictionary
    estimators = [
        (name, results["best_estimator"]) for name, results in cv_results.items()
    ]

    # Variables to store of the best model
    best_model = None
    best_accuracy = 0
    best_estimators = None
    best_predicted_labels = None
    best_predicted_prob = None
    best_metrics = None

    # Iterate through all combinations of estimators
    for i in range(1, len(estimators) + 1):
        for comb in combinations(estimators, i):
            comb = list(comb)

            # Perform StackingClassification
            stacking_classifier = StackingClassifier(
                estimators=comb,
                final_estimator=LogisticRegression(
                    **cv_results["Logistic Regression"]["best_params"]
                ),
                cv=5,
                n_jobs=-1,
            )
            stacking_classifier.fit(X_train, Y_train)
            stacking_pred = stacking_classifier.predict(X_test)
            stacking_accuracy = model_accuracy(stacking_pred)
            stacking_prob = predicted_prob(stacking_classifier)

            # Compare if this is the best model
            if stacking_accuracy > best_accuracy:
                best_model = stacking_classifier
                best_accuracy = stacking_accuracy
                best_estimators = comb
                best_predicted_labels = stacking_pred
                best_predicted_prob = stacking_prob
                best_metrics = compute_metrics(Y_test, stacking_pred)

            # Perform VotingClassification
            voting_classifier = VotingClassifier(
                estimators=comb, voting="soft", n_jobs=-1
            )
            voting_classifier.fit(X_train, Y_train)
            voting_pred = voting_classifier.predict(X_test)
            voting_accuracy = model_accuracy(voting_pred)
            voting_prob = predicted_prob(voting_classifier)

            # Compare if this is the best model
            if voting_accuracy > best_accuracy:
                best_model = voting_classifier
                best_accuracy = voting_accuracy
                best_estimators = comb
                best_predicted_labels = voting_pred
                best_predicted_prob = voting_prob
                best_metrics = compute_metrics(Y_test, voting_pred)

    # Return the best model, its accuracy, and other details
    return {
        "model": best_model,
        "accuracy": best_accuracy,
        "estimators": best_estimators,
        "predicted_labels": best_predicted_labels,
        "predicted_prob": best_predicted_prob,
        "metrics": best_metrics,
    }


def evaluate_ensemble_model(
    ensemble_model: StackingClassifier | VotingClassifier,
    df: pd.DataFrame,
    label_col: str,
    n_splits: int = 10,
    random_state: int | None = 42,
) -> dict:
    """
    Evaluate the performance of an ensemble model using k-fold cross-validation.

    Args:
        (StackingClassifier | VotingClassifier): The ensemble model to be evaluated.
        pd.DataFrame: The dataset containing features and the target label column.
        label_col str: The name of the column containing the target labels.
        (int, optional): The number of folds for stratified k-fold CV. (Default is 10)
        (int | None, optional): Random seed for reproducibility. (Default is 42)

    Returns:
        dict: A dictionary containing the following evaluation statistics:
            - Accuracy: Tuple of mean and std of accuracy scores across folds.
            - F1-Score: Tuple of mean and std of F1-scores across folds.
            - Precision: Tuple of mean and std of precision scores across folds.
            - Recall: Tuple of mean and std of recall scores across folds.
            - Confusion Matrix Normalized cumulative confusion matrix across all folds.
            - True Labels List of true labels from all folds.
            - Predicted Probabilities: List of predicted probabilities for
                                       the positive class from all folds.
    """
    X = df.drop(columns=label_col)
    Y = df[label_col]
    k_fold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    (
        accuracy_scores,
        all_probabilities,
        all_true_labels,
        f1_scores,
        precision_scores,
        recall_scores,
    ) = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    cumulative_confusion_matrix = np.zeros((2, 2), dtype=int)
    for fold, (train_idx, test_idx) in enumerate(k_fold.split(X, Y), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]
        ensemble_model.fit(X_train, Y_train)
        Y_pred = ensemble_model.predict(X_test)
        Y_prob = ensemble_model.predict_proba(X_test)[:, 1]
        accuracy_scores.append(accuracy_score(Y_test, Y_pred))
        all_true_labels.extend(Y_test)
        all_probabilities.extend(Y_prob)
        f1_scores.append(f1_score(Y_test, Y_pred))
        precision_scores.append(precision_score(Y_test, Y_pred))
        recall_scores.append(recall_score(Y_test, Y_pred))
        cumulative_confusion_matrix += confusion_matrix(Y_test, Y_pred)
    return {
        "Accuracy": (np.mean(accuracy_scores), np.std(accuracy_scores)),
        "F1-Score": (np.mean(f1_scores), np.std(f1_scores)),
        "Precision": (np.mean(precision_scores), np.std(precision_scores)),
        "Recall": (np.mean(recall_scores), np.std(recall_scores)),
        "Confusion Matrix": cumulative_confusion_matrix
        / np.sum(cumulative_confusion_matrix),
        "True Labels": all_true_labels,
        "Predicted Probabilities": all_probabilities,
    }


def evaluate_predictor_accuracy(
    ensemble_model: StackingClassifier | VotingClassifier,
    X_test: pd.DataFrame,
    Y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate the accuracy of each individual predictor (base estimator)
    in the ensemble model and represent the results as a DataFrame.

    Args:
        (StackingClassifier | VotingClassifier): The ensemble model.
        pd.DataFrame: The testing data features.
        pd.Series: The testing data labels.

    Returns:
        pd.DataFrame: A DataFrame with each predictor's name and its accuracy score.

    Raises:
        ValueError: If an unsupported ensemble model is passed.
    """
    # Get the base estimator depending on the ensemble type
    if isinstance(ensemble_model, VotingClassifier):
        base_estimators = ensemble_model.estimators
    elif isinstance(ensemble_model, StackingClassifier):
        base_estimators = ensemble_model.named_estimators_.items()
    else:
        raise ValueError("Unsupported ensemble model type.")

    feature_accuracies = []
    for name, estimator in base_estimators:
        Y_pred = estimator.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)
        feature_accuracies.append({"Estimator": name, "Accuracy": accuracy})

    return pd.DataFrame(feature_accuracies)


def plot_ensemble_evaluation(results: dict, data_name: str, n_bins: int = 10) -> None:
    """
    Function to plot the evaluation metrics and visualizations for an ensemble model.

    Args:
        dict: The evaluation results dictionary containing metrics and confusion matrix.
        str: The name of the data (e.g., 'BASE') to be displayed in the plot titles.
        int: Number of bins to discretize the [0, 1] interval for the calibration curve.
             (Default is 10)

    Returns:
        None
    """
    # Create subplots with 2 rows and 2 columns
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    colors = ["red", "blue", "skyblue", "orange", "green", "yellow", "black"]

    # Plot the confusion matrix on the first subplot (ax1)
    sns.heatmap(
        results["Confusion Matrix"] * 100,
        annot=True,
        fmt=".1f",
        cmap="inferno_r",
        xticklabels=["Stay In MCI", "Convert To AD"],
        yticklabels=["Stay In MCI", "Convert To AD"],
        annot_kws={"size": 11},
        ax=ax1,
    )
    for text in ax1.texts:
        text.set_text(f"{text.get_text()}%")
    ax1.set_title(f"Confusion Matrix [{data_name}]")
    ax1.set_xlabel("Predicted Label")
    ax1.set_ylabel("True Label")

    # Plot the ROC curve on the second subplot (ax2)
    true_labels = results["True Labels"]
    predicted_probabilities = results["Predicted Probabilities"]
    fpr, tpr, _ = roc_curve(true_labels, predicted_probabilities)
    roc_auc = auc(fpr, tpr)
    ax2.plot(
        fpr,
        tpr,
        color=random.choice(colors),
        label=f"Area Under Curve = {roc_auc:.2f}",
        linewidth=3,
    )
    ax2.plot([0, 1], [0, 1], color=random.choice(colors), linestyle="--")
    ax2.set_title(f"ROC Curve [{data_name}]")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.375)

    # Plot the metrics bar chart on ax3
    metrics = ["Accuracy", "F1-Score", "Precision", "Recall"]
    mean_vals = [results[metric][0] for metric in metrics]
    std_vals = [results[metric][1] for metric in metrics]
    bars = ax3.bar(
        metrics, mean_vals, yerr=std_vals, capsize=5, color=random.choice(colors)
    )
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("Metrics")
    ax3.set_ylabel("Score")
    ax3.set_title(f"Evaluation Metrics with Standard Deviations [{data_name}]")
    legend_labels = [
        f"{metric}: {mean:.2f} ± {std:.2f}"
        for metric, mean, std in zip(metrics, mean_vals, std_vals)
    ]
    ax3.legend(bars, legend_labels, loc="lower right")

    # Plot the calibration curve on ax4
    fraction_of_positives, mean_predicted_value = calibration_curve(
        results["True Labels"],
        results["Predicted Probabilities"],
        n_bins=10,
    )
    ax4.plot(
        mean_predicted_value,
        fraction_of_positives,
        marker="o",
        label="Calibration Curve",
        color=random.choice(colors),
        linewidth=3,
    )
    ax4.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfectly Calibrated",
        color=random.choice(colors),
        linewidth=3,
    )
    ax4.set_title(f"Calibration Curve For Ensemble Model [{data_name}]")
    ax4.set_xlabel("Mean Predicted Value")
    ax4.set_ylabel("Fraction Of Positives")
    ax4.legend()
    ax4.grid(alpha=0.375)

    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.tight_layout()
    plt.show()


def pretty_print_ensemble_model_results(data_set_name: str, results: dict) -> None:
    """
    Prints the results of the model evaluation in a nicely formatted way.

    Args:
        str: The name of the CSV that the model was applied on.
        dict: The output of the `evaluate_ensemble_model` function.

    Returns:
        None
    """
    results_copy = results.copy()
    results_copy.pop("Confusion Matrix")
    results_copy.pop("True Labels")
    results_copy.pop("Predicted Probabilities")
    print(f"Results for the {data_set_name} dataset using the ensemble model -")
    for idx, (key, value) in enumerate(results_copy.items(), start=1):
        print(f"{idx}. {key}:\t {round(value[0], 2)} ± {round(value[1], 2)}")


def split_data(
    df: pd.DataFrame,
    label_col: str,
    test_size: float = 0.2,
    random_state: int | None = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a DataFrame into training and testing data/labels.

    Args:
        pd.DataFrame: The DataFrame to split.
        str: Name of the column to use as labels.
        float: The percentage of the dataset to use in the test split. (Default is 0.2)
        (int | None, optional): Random seed for reproducibility. (Default is 42)

    Returns:
        (X_train, X_test, Y_train, Y_test): The data and labels split into
                                            training and testing datasets.
    """
    # Remove the column that will be used as the label
    X = df.drop(columns=label_col)
    Y = df[label_col]

    # Split the dataset into train/test
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, Y_train, Y_test


def over_sample_data(
    method: Over_samplers,
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    random_state: int | None = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Performs oversampling on the training data using the specified method.

    Args:
        Over_samplers: The oversampling method to use. Must be one of:
            - Over_samplers.ADASYN: Adaptive Synthetic Sampling.
            - Over_samplers.RANDOM: Random Oversampling.
            - Over_samplers.SMOTE:  Synthetic Minority Oversampling Technique.
        pd.DataFrame: The training feature data.
        pd.Series: The training target labels.
        (int | None, optional): Random seed for reproducibility. (Default is 42)

    Returns:
        (tuple[pd.DataFrame, pd.Series]): A tuple containing:
            - pd.DataFrame: The resampled feature data.
            - pd.Series: The resampled target labels.

    Raises:
        ValueError: If an unsupported oversampling method is provided.
    """
    match method:
        case Over_samplers.ADASYN:
            model = ADASYN(random_state=random_state)
        case Over_samplers.RANDOM:
            model = RandomOverSampler(random_state=random_state)
        case Over_samplers.SMOTE:
            model = SMOTE(random_state=random_state)
        case _:
            raise ValueError("Unsupported Oversampling Method")
    X_train_resampled, Y_train_resampled = model.fit_resample(X_train, Y_train)
    return X_train_resampled, Y_train_resampled


def tune_hyperparameters(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_train: pd.Series,
    Y_test: pd.Series,
    random_state: int | None = 42,
) -> dict:
    """
    Perform hyperparameter tuning for multiple
    machine learning models using GridSearchCV.

    Args:
        pd.DataFrame: The training data features.
        pd.DataFrame: The testing data features.
        pd.Series: The training data labels.
        pd.Series: The testing data labels.
        (int | None, optional): Random seed for reproducibility. (Default is 42)

    Returns:
        dict: A dictionary containing the following information for each model:
            - "cv_results" (dict): Detailed results from GridSearchCV.
            - "best_params" (dict): Best hyperparameter combination found.
            - "best_f1_score" (float): Best F1 score achieved during CV.
            - "best_accuracy" (float): Accuracy score of the best estimator on test data.
            - "best_estimator" (object): The best trained model for each classifier.
            - "predicted_labels" (ndarray): Predicted labels for the test data.
            - "predicted_prob" (ndarray): Predicted probabilities for the test set.
    """
    # [0.001, 0.01, 0.1, 1, 10, 10]
    c = [10**i for i in range(-3, 2)]
    model_with_params = {
        "KNN": {
            "Model": KNeighborsClassifier(),
            "Parameters": {
                "n_neighbors": list(range(1, 16)),
                "weights": ["uniform", "distance"],
            },
        },
        "Logistic Regression": {
            "Model": LogisticRegression(random_state=random_state),
            "Parameters": {
                "C": c,
                "solver": [
                    "lbfgs",
                    "liblinear",
                    "newton-cg",
                    "newton-cholesky",
                    "sag",
                    "saga",
                ],
            },
        },
        "SVM": {
            "Model": SVC(random_state=random_state),
            "Parameters": {
                "C": c,
                "kernel": ["linear", "poly", "rbf", "sigmoid"],
            },
        },
        "MLP": {
            "Model": MLPClassifier(max_iter=500, random_state=random_state),
            "Parameters": {
                "activation": ["relu", "identity", "logistic", "tanh"],
                "hidden_layer_sizes": [(50,), (100,), (50, 50)],
            },
        },
        "Random Forest": {
            "Model": RandomForestClassifier(random_state=random_state),
            "Parameters": {
                "n_estimators": list(range(50, 201, 30)),
                "max_depth": [None] + list(range(10, 51, 10)),
            },
        },
        "Decision Tree": {
            "Model": DecisionTreeClassifier(random_state=random_state),
            "Parameters": {
                "criterion": ["gini", "entropy", "log_loss"],
                "max_depth": [None] + list(range(5, 31, 5)),
                "max_features": ["auto", "sqrt", "log2"],
            },
        },
    }

    scoring = {
        "f1_score": "f1_weighted",
        "accuracy": make_scorer(accuracy_score),
        "precision": "precision_weighted",
        "recall": "recall_weighted",
    }

    # Apply GridSearchCV for each model
    cv_results = {}
    CV = 5
    for name, model_with_params in model_with_params.items():
        grid = GridSearchCV(
            model_with_params["Model"],
            model_with_params["Parameters"],
            cv=CV,
            scoring=scoring,
            refit="f1_score",
            n_jobs=-1,
            verbose=False,
        )
        grid.fit(X_train, Y_train)
        best_model = grid.best_estimator_

        # Special handling for SVM, wrap with CalibratedClassifierCV
        if name == "SVM":
            best_model = CalibratedClassifierCV(best_model, cv=CV)
            best_model.fit(X_train, Y_train)

        Y_pred = best_model.predict(X_test)

        # Get probabilities or decision scores
        if hasattr(best_model, "predict_proba"):
            Y_prob = best_model.predict_proba(X_test)[:, 1]
        elif hasattr(best_model, "decision_function"):
            Y_prob = best_model.decision_function(X_test)
        else:
            Y_prob = None

        cv_results[name] = {
            "cv_results": grid.cv_results_,
            "best_params": grid.best_params_,
            "best_f1_score": grid.best_score_,
            "best_accuracy": accuracy_score(Y_test, Y_pred),
            "best_estimator": best_model,
            "predicted_labels": Y_pred,
            "predicted_prob": Y_prob,
        }
    return cv_results
