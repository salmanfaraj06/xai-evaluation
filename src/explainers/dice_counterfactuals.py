"""DiCE counterfactual wrapper using processed feature space."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


class DiceExplainer:
    def __init__(
        self,
        model,
        X_train_processed: np.ndarray,
        y_train,
        feature_names: List[str],
        outcome_name: str,
        method: str = "random",
    ):
        try:
            import dice_ml  # type: ignore
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("Install dice-ml to use DiceExplainer") from exc

        self.feature_names = feature_names

        df = pd.DataFrame(X_train_processed, columns=feature_names)
        df[outcome_name] = y_train.values

        data = dice_ml.Data(
            dataframe=df,
            continuous_features=feature_names,
            outcome_name=outcome_name,
        )
        model_wrapper = dice_ml.Model(
            model=model,
            backend="sklearn",
            model_type="classifier",
        )
        self.dice = dice_ml.Dice(data, model_wrapper, method=method)
        self.outcome_name = outcome_name

    def generate_counterfactuals(
        self,
        x_row_processed: np.ndarray,
        total_cfs: int = 3,
    ):
        """Generate counterfactuals in processed feature space."""
        query = pd.DataFrame(
            [x_row_processed],
            columns=self.feature_names,
        )
        return self.dice.generate_counterfactuals(
            query,
            total_CFs=total_cfs,
            desired_class="opposite",
            verbose=False,
        )
