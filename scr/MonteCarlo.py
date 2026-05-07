import pandas as pd
import numpy as np
import scipy.stats as stats
import warnings

# Suppress math warnings when exotic curves fail to fit
warnings.filterwarnings('ignore')

print("--- Phase 2: The Enterprise 35-Curve Tournament ---")

# ==========================================
# 1. LOAD AND CLEAN
# ==========================================
raw_data = pd.read_csv('data/prodliab_pos_98-07.csv')
company_data = raw_data[raw_data['GRNAME'] == 'Allstate Ins Co Grp'].copy()
company_data = company_data.sort_values(['AccidentYear', 'DevelopmentLag'])
company_data['IncrementalPaid'] = company_data.groupby('AccidentYear')['CumPaidLoss'].diff()
company_data['IncrementalPaid'] = company_data['IncrementalPaid'].fillna(company_data['CumPaidLoss'])
clean_data = company_data[company_data['IncrementalPaid'] > 0]
historical_losses = clean_data['IncrementalPaid'].values

# ==========================================
# 2. THE 35 DISTRIBUTION TEST
# ==========================================
print(f"Testing 35 actuarial distributions against {len(historical_losses)} historical records...\n")

# A massive list of standard and exotic statistical curves
dist_names = [
    'lognorm', 'gamma', 'weibull_min', 'genpareto', 'expon', 'invgauss', 
    'fisk', 'burr', 'pearson3', 'loggamma', 'halfcauchy', 'gumbel_r', 
    'logistic', 'maxwell', 'rayleigh', 'laplace', 't', 'chi2', 'wald', 
    'nakagami', 'genextreme', 'fatiguelife', 'foldnorm', 'halfnorm', 
    'levy', 'lomax', 'mielke', 'kappa4', 'johnsonsu', 'johnsonsb', 
    'gompertz', 'gilbrat', 'erlang', 'exponnorm', 'dweibull'
]

results = []

for name in dist_names:
    try:
        # Fetch the mathematical function from SciPy
        dist = getattr(stats, name)
        
        # Fit the distribution to our data
        params = dist.fit(historical_losses)
        
        # Calculate AIC (Akaike Information Criterion)
        log_likelihood = np.sum(dist.logpdf(historical_losses, *params))
        k = len(params)
        aic = 2 * k - 2 * log_likelihood
        
        # Calculate KS-Stat
        D, p_value = stats.kstest(historical_losses, name, args=params)
        
        # Ensure the math worked before appending
        if not np.isnan(aic) and not np.isinf(aic):
            results.append({
                'Distribution': name,
                'AIC': aic,
                'KS_Stat': D,
                'Params': params,
                'Dist_Object': dist
            })
    except Exception as e:
        pass # Skip curves that mathematically crash on this specific dataset

# Sort by AIC (Lowest is Best)
leaderboard = pd.DataFrame(results).sort_values(by='AIC').reset_index(drop=True)

print("🏆 --- TOP 5 DISTRIBUTIONS (Ranked by AIC) --- 🏆")
print(leaderboard[['Distribution', 'AIC', 'KS_Stat']].head(5).to_string(index=False))

# ==========================================
# 3. MONTE CARLO WITH THE CHAMPION
# ==========================================
winner_row = leaderboard.iloc[0]
best_name = winner_row['Distribution']
best_params = winner_row['Params']
best_dist = winner_row['Dist_Object']

print(f"\n✅ WINNER SELECTED: {best_name.upper()}")
print(f"Executing 10,000 Monte Carlo simulations using {best_name} logic...")

iterations = 10000
claims_per_year = 10
simulated_years = []

for _ in range(iterations):
    # Simulate claims using the winning curve's specific shape
    simulated_claims = best_dist.rvs(*best_params, size=claims_per_year)
    simulated_claims = np.maximum(simulated_claims, 0) # No negative claims
    simulated_years.append(np.sum(simulated_claims))

simulated_years = np.array(simulated_years)

expected_loss = np.mean(simulated_years)
value_at_risk_99 = np.percentile(simulated_years, 99)

print("\n--- OPTIMIZED STRESS TEST RESULTS ---")
print(f"Average Expected Yearly Loss:   ${expected_loss * 1000:,.2f}")
print(f"Worst-Case Scenario (99% VaR):  ${value_at_risk_99 * 1000:,.2f}")
print(f"Capital Buffer Required:        ${(value_at_risk_99 - expected_loss) * 1000:,.2f}")