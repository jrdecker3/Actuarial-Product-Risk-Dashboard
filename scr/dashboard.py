import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# --- 1. PAGE SETUP ---
st.set_page_config(page_title="Actuarial Risk & Reserving", layout="wide", page_icon="📊")
st.title("📊 Enterprise Risk & Reserving Dashboard")
st.markdown("Multi-Tenant Workstation: Stochastic Pricing & Loss Development Triangles")

# --- 2. DATA LOADING ---
@st.cache_data
def load_master_data():
    return pd.read_csv('data/cleaned_cas_data.csv')

try:
    df = load_master_data()
except FileNotFoundError:
    st.error("Could not find 'cleaned_cas_data.csv'. Please run the clean_data.py script first.")
    st.stop()

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("🏢 Portfolio Selection")
company_list = sorted(df['GRNAME'].unique())
selected_company = st.sidebar.selectbox("Select Insurance Carrier:", company_list)

st.sidebar.markdown("---")
st.sidebar.header("🎲 Simulation Parameters")
iterations = st.sidebar.number_input("Monte Carlo Iterations:", min_value=1000, max_value=50000, value=10000, step=1000)
expected_claims = st.sidebar.slider("Expected Claims/Year (Frequency):", min_value=1, max_value=100, value=15)

# Extract data for the selected company
company_data = df[df['GRNAME'] == selected_company].sort_values(['AccidentYear', 'DevelopmentLag'])
historical_losses = company_data['IncrementalPaid'].values

# --- 4. CALCULUS ENGINE (Tournament of Curves) ---
@st.cache_data(show_spinner=False)
def fit_distributions(data_array):
    dist_names = [
        'weibull_min', 'gamma', 'lognorm', 'expon', 'genpareto', 'invgauss', 
        'fisk', 'burr', 'pearson3', 'loggamma', 'gumbel_r', 'logistic', 'laplace'
    ]
    results = []
    for name in dist_names:
        try:
            dist = getattr(stats, name)
            params = dist.fit(data_array)
            log_likelihood = np.sum(dist.logpdf(data_array, *params))
            k = len(params)
            aic = 2 * k - 2 * log_likelihood
            if not np.isnan(aic) and not np.isinf(aic):
                results.append({'Distribution': name, 'AIC': aic, 'Params': params, 'Dist_Object': dist})
        except Exception:
            pass
    return pd.DataFrame(results).sort_values(by='AIC').reset_index(drop=True)

with st.spinner(f"Evaluating distribution fits for {selected_company}..."):
    leaderboard = fit_distributions(historical_losses)

# --- TABS SETUP ---
tab1, tab2 = st.tabs(["🎲 Stochastic Simulation", "🔺 Loss Development Triangles"])

# ==========================================
# TAB 1: STOCHASTIC SIMULATION
# ==========================================
with tab1:
    st.header("Monte Carlo Severity & Frequency Engine")
    
    col_ctrl, col_lead = st.columns([1, 2])
    with col_ctrl:
        st.subheader("Distribution Override")
        st.markdown("The AI has selected the Best Fit based on the lowest AIC score. You may override this selection on the fly.")
        # Change distribution on the fly
        selected_dist_name = st.selectbox(
            "Select Active Distribution:", 
            leaderboard['Distribution'].tolist(),
            index=0 # Defaults to the #1 Best Fit
        )
        
    with col_lead:
        st.subheader("🏆 Goodness-of-Fit Leaderboard")
        st.dataframe(leaderboard[['Distribution', 'AIC']].head(5), use_container_width=True)

    # Grab the parameters for the user's selected distribution
    active_row = leaderboard[leaderboard['Distribution'] == selected_dist_name].iloc[0]
    active_dist = active_row['Dist_Object']
    active_params = active_row['Params']

    # Run Monte Carlo
    @st.cache_data(show_spinner=False)
    def run_simulation(dist_name, _dist, params, expected_freq, iters):
        freqs = np.random.poisson(lam=expected_freq, size=iters)
        sims = []
        for f in freqs:
            if f == 0:
                sims.append(0)
            else:
                claims = _dist.rvs(*params, size=f)
                claims = np.maximum(claims, 0)
                sims.append(np.sum(claims))
        return np.array(sims)

    sim_results = run_simulation(selected_dist_name, active_dist, active_params, expected_claims, iterations)

    st.markdown("---")
    # Two-Tail Range Analysis
    st.subheader("🎯 Value at Risk (VaR) & Range Analysis")
    
    max_sim = float(np.max(sim_results))
    val_95 = float(np.percentile(sim_results, 95))
    
    range_vals = st.slider("Select Loss Threshold Range:", 0.0, max_sim, (0.0, val_95), step=100.0, format="$%f")
    
    prob_in = np.mean((sim_results >= range_vals[0]) & (sim_results <= range_vals[1])) * 100
    prob_below = np.mean(sim_results < range_vals[0]) * 100
    prob_above = np.mean(sim_results > range_vals[1]) * 100

    m1, m2, m3 = st.columns(3)
    m1.info(f"**Prob < ${range_vals[0]:,.0f}:** \n\n {prob_below:.2f}%")
    m2.success(f"**Prob in Range:** \n\n {prob_in:.2f}%")
    m3.error(f"**Prob > ${range_vals[1]:,.0f} (Tail Risk):** \n\n {prob_above:.2f}%")

    # Histogram
    fig, ax = plt.subplots(figsize=(10, 4))
    n, bins, patches = ax.hist(sim_results, bins=100, color='lightgray', edgecolor='black', alpha=0.7)
    for i in range(len(bins)-1):
        if range_vals[0] <= bins[i] <= range_vals[1]:
            patches[i].set_facecolor('royalblue')
    
    ax.axvline(np.percentile(sim_results, 99), color='red', linestyle='dashed', lw=2, label='99% VaR')
    ax.set_title(f"Simulated Annual Losses ({selected_dist_name.upper()})")
    ax.set_xlabel("Total Annual Loss ($)")
    ax.legend()
    st.pyplot(fig)


# ==========================================
# TAB 2: LOSS DEVELOPMENT TRIANGLES
# ==========================================
with tab2:
    st.header("Actuarial Loss Development Triangles")
    st.markdown("NAIC Schedule P formatting. Tracks how claims from a specific Accident Year grow over time (Development Lag).")
    
    # Generate the Pivot Table (Triangle)
    # Rows = Accident Year, Columns = Lag, Values = Cumulative Paid Loss
    triangle = company_data.pivot_table(
        index='AccidentYear', 
        columns='DevelopmentLag', 
        values='CumPaidLoss', 
        aggfunc='sum'
    )
    
    st.subheader("Cumulative Paid Loss Triangle ($)")
    # We use Pandas Styling to add a background gradient (heatmap) to the triangle
    st.dataframe(triangle.style.background_gradient(cmap='Blues', axis=None).format("{:,.0f}", na_rep="-"), use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Development Curves")
    st.markdown("Visualizing the trajectory of claim maturation by Accident Year.")
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    # Plot each accident year as a line
    for acc_year in triangle.index:
        ax2.plot(triangle.columns, triangle.loc[acc_year], marker='o', label=f'AY {acc_year}')
    
    ax2.set_xlabel("Development Lag (Years)")
    ax2.set_ylabel("Cumulative Paid Loss ($)")
    ax2.set_title("Loss Development Trajectories")
    # Only show legend if there aren't too many years
    if len(triangle.index) <= 15:
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig2)