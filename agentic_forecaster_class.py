"""
Agentic forecaster: class-based prompt-driven forecasting using sktime.
"""

import re
from sktime.datasets import load_airline
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.base import ForecastingHorizon


class AgenticForecaster:
    def __init__(self):
        self.model = ThetaForecaster(sp=12)
        self.data = load_airline()
        self.model.fit(self.data)

    def _extract_steps(self, prompt):
        match = re.search(r"\d+", prompt)
        return int(match.group()) if match else 3

    def _analyze_trend(self, predictions):
        if predictions.iloc[-1] > self.data.iloc[-1]:
            return "increasing"
        return "decreasing or stable"

    def predict_from_prompt(self, prompt):
        steps = self._extract_steps(prompt)

        fh = ForecastingHorizon(list(range(1, steps + 1)), is_relative=True)
        preds = self.model.predict(fh)

        explanation = None
        if "explain" in prompt.lower() or "trend" in prompt.lower():
            trend = self._analyze_trend(preds)
            explanation = (
                f"The forecast suggests an overall {trend} trend "
                f"for the next {steps} steps."
            )

        return preds, explanation


# Example usage
if __name__ == "__main__":
    agent = AgenticForecaster()

    prompt = input("Enter your prompt: ")
    preds, explanation = agent.predict_from_prompt(prompt)

    print("Prompt:", prompt)
    print("\nPredictions:\n", preds)
    print("\nExplanation:", explanation)