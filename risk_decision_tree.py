import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.tree import DecisionTreeClassifier
import os
import warnings

warnings.filterwarnings('ignore')

class ReinsuranceEngine:
    def __init__(self):
        # File paths
        self.capital_path = os.path.join('data', 'carrier_capital.csv')
        self.claims_path = os.path.join('data', 'cleaned_cas_data.csv') # Ensure this matches your historical data file
        
        # Initialize the Machine Learning model
        self.decision_tree = self._train_reinsurance_tree()

    def _train_reinsurance_tree(self):
        """
        Trains a Decision Tree Classifier based on Actuarial Logic.
        Features: [Loss_to_Surplus_Ratio, Frequency_Severity_Ratio]
        Labels: 0 (Retain), 1 (Quota Share), 2 (Excess of Loss), 3 (Facultative)
        """
        # Synthetic training data representing real-world underwriting guidelines
        X = np.array([
            [0.05, 0.1], [0.08, 0.2],  # Low loss ratio, low severity -> Retain
            [0.15, 0.8], [0.20, 0.9],  # Medium loss ratio, high frequency -> Quota Share
            [0.40, 0.3], [0.60, 0.2],  # High loss ratio, massive single severity -> Excess of Loss
            [0.85, 0.9], [0.95, 0.8]   # Company killer (High ratio, high severity) -> Facultative
        ])
        y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        
        tree = DecisionTreeClassifier(max_depth=3, random_state=42)
        tree.fit(X, y)
        return tree

    def load_data(self):
        """Loads the baseline capital and the historical claims."""
        if not os.path.exists(self.capital_path):
            raise FileNotFoundError("Run capital_engine.py first to generate carrier_capital.csv")
        
        self.capital_df = pd.read_csv(self.capital_path)
        
        # Placeholder for historical claims (loads the Allstate data you cleaned earlier)
        # If your file is named differently, update self.claims_path in __init__
        try:
            self.claims_df = pd.read_csv(self.claims_path)
            self.historical_losses = self.claims_df['IncrementalPaid'].dropna().values
        except FileNotFoundError:
            # Fallback synthetic data if the CSV isn't found during testing
            print("Historical claims not found. Using synthetic severity distribution.")
            self.historical_losses = stats.weibull_min.rvs(1.5, scale=5000000, size=1000)

    def run_monte_carlo(self, iterations=10000):
        """Fits a curve to historical data and simulates 10,000 future scenarios."""
        print("Fitting severity curve and running Monte Carlo simulation...")
        
        # Fit a Log-Normal distribution to the claims
        shape, loc, scale = stats.lognorm.fit(self.historical_losses, floc=0)
        
        # Simulate future scenarios
        simulated_losses = stats.lognorm.rvs(shape, loc=loc, scale=scale, size=iterations)
        
        # Calculate Value at Risk (99th percentile worst-case)
        var_99 = np.percentile(simulated_losses, 99)
        return var_99

    def evaluate_carrier(self, ticker):
        """Evaluates a specific carrier's capital against the simulated risk."""
        carrier_info = self.capital_df[self.capital_df['Ticker'] == ticker.upper()]
        
        if carrier_info.empty:
            return f"Error: Ticker {ticker} not found in capital database."
            
        surplus_billions = carrier_info['Surplus_Billions'].values[0]
        surplus_actual = surplus_billions * 1_000_000_000
        
        # Get the simulated VaR
        var_99_loss = self.run_monte_carlo()
        
        # Feature Engineering for the Decision Tree
        loss_ratio = var_99_loss / surplus_actual
        severity_ratio = 0.5 # Placeholder for standard product liability severity index
        
        # Predict the Reinsurance Structure
        features = np.array([[loss_ratio, severity_ratio]])
        prediction = self.decision_tree.predict(features)[0]
        
        treaty_types = {
            0: "Net Retention (No Reinsurance Needed)",
            1: "Quota Share Treaty",
            2: "Excess of Loss (XoL) Treaty",
            3: "Facultative Placement (Extreme Risk)"
        }
        
        # Calculate the "Burning Cost" (Pure Premium for the Reinsurance Layer)
        # Assumes a 15% risk load
        recommended_premium = var_99_loss * 0.01 * 1.15 if prediction > 0 else 0

        # Construct Output
        report = {
            "Carrier": carrier_info['Company'].values[0],
            "Available Surplus": f"${surplus_actual:,.2f}",
            "1-in-100 Year Loss (VaR)": f"${var_99_loss:,.2f}",
            "Loss-to-Surplus Ratio": f"{loss_ratio * 100:.4f}%",
            "Recommended Strategy": treaty_types[prediction],
            "Est. Reinsurance Premium": f"${recommended_premium:,.2f}"
        }
        
        return report

if __name__ == "__main__":
    engine = ReinsuranceEngine()
    engine.load_data()
    
    # Test the model on Allstate (ALL)
    print("\n--- Actuarial Risk & Capital Report ---")
    result = engine.evaluate_carrier('ALL')
    for key, value in result.items():
        print(f"{key}: {value}")