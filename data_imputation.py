from dataclasses import dataclass
from typing import Union


@dataclass
class Knn:
    """
    Represents a K-Nearest Neighbors (KNN) imputer configuration.

    Attributes:
        n (int): The number of nearest neighbors to use in imputation.
    """

    n: int


@dataclass
class Mice_forest:
    """
    Represents a MICE (Multiple Imputation by Chained Equations) forest
    imputer configuration.

    Attributes:
        iterations (int): The number of iterations for the MICE process.
        num_datasets (int): The number of imputed datasets to generate,
                            providing multiple complete data sets for
                            further analysis or aggregation.
    """

    iterations: int
    num_datasets: int


Imputation_type = Union[Knn, Mice_forest]
