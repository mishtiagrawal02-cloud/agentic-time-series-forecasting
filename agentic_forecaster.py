import re
from sktime.datasets import load_airline
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from sklearn.metrics import mean_absolute_error


class AgenticForecaster:
    def __init__(self):
        self.data = load_airline()

    def _extract_steps(self, prompt):
        match = re.search(r"\d+", prompt)
        return int(match.group()) if match else 3

    def _select_model(self, prompt):
        prompt = prompt.lower()

        if "smooth" in prompt:
            return ExponentialSmoothing()

        elif "naive" in prompt:
            return NaiveForecaster(strategy="last")

        elif "fast" in prompt:
            return ThetaForecaster(sp=12)

        return ThetaForecaster(sp=12)

    def _analyze_trend(self, predictions):
        if predictions.iloc[-1] > self.data.iloc[-1]:
            return "increasing"
        elif predictions.iloc[-1] < self.data.iloc[-1]:
            return "decreasing"
        else:
            return "stable"

    def _evaluate_models(self, steps):
        models = {
            "Theta": ThetaForecaster(sp=12),
            "Naive": NaiveForecaster(strategy="last"),
            "ExpSmoothing": ExponentialSmoothing(),
        }

        results = {}

        for name, model in models.items():
            model.fit(self.data)

            fh = ForecastingHorizon(list(range(1, steps + 1)), is_relative=True)
            preds = model.predict(fh)

            actual = self.data.iloc[-steps:]
            preds = preds.iloc[:len(actual)]

            error = mean_absolute_error(actual, preds)
            results[name] = (model, error)

        best_model_name = min(results, key=lambda x: results[x][1])
        return results[best_model_name][0], best_model_name, results

    def predict_from_prompt(self, prompt):
        steps = self._extract_steps(prompt)

        # Agent decision
        if "best" in prompt.lower() or "compare" in prompt.lower():
            self.model, best_name, results = self._evaluate_models(steps)
        else:
            self.model = self._select_model(prompt)

        self.model.fit(self.data)

        fh = ForecastingHorizon(list(range(1, steps + 1)), is_relative=True)
        preds = self.model.predict(fh)

        explanation = None
        if "explain" in prompt.lower() or "trend" in prompt.lower():
            trend = self._analyze_trend(preds)

            if "best" in prompt.lower() or "compare" in prompt.lower():
                explanation = (
                    f"Compared multiple models and selected {best_name} as best "
                    f"based on lowest forecasting error."
                )
            else:
                explanation = (
                    f"Using {self.model.__class__.__name__}, "
                    f"the model predicts values from {preds.iloc[0]:.2f} "
                    f"to {preds.iloc[-1]:.2f}, indicating a {trend} trend "
                    f"compared to the last observed value {self.data.iloc[-1]:.2f}."
                )

        return preds, explanation


if __name__ == "__main__":
    agent = AgenticForecaster()

    prompt = input("Enter your prompt: ")
    preds, explanation = agent.predict_from_prompt(prompt)

    print("\nPredictions:\n", preds)
    print("\nExplanation:", explanation)