from matplotlib.pylab import rand
import pandas as pd
from enum import Enum
from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class Over_samplers(Enum):
    RANDOM = "Random"
    SMOTE = "SMOTE"
    ADASYN = "ADASYN"


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
) -> dict[dict, dict, float, float]:
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
    for name, model_with_params in model_with_params.items():
        grid = GridSearchCV(
            model_with_params["Model"],
            model_with_params["Parameters"],
            cv=5,
            scoring=scoring,
            refit="f1_score",
            n_jobs=-1,
            verbose=False,
        )
        grid.fit(X_train, Y_train)
        cv_results[name] = {
            "cv_results": grid.cv_results_,
            "best_params": grid.best_params_,
            "best_f1_score": grid.best_score_,
            "best_accuracy": accuracy_score(
                Y_test, grid.best_estimator_.predict(X_test)
            ),
        }
    return cv_results
