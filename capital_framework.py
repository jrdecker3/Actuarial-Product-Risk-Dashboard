import yfinance as yf
import pandas as pd
import os

def fetch_carrier_capital():
    # sample of the top publicly traded P&C and Reinsurance carriers
    tickers = {
        'ALL': 'Allstate',
        'TRV': 'Travelers',
        'CB': 'Chubb',
        'PGR': 'Progressive',
        'AIG': 'AIG',
        'HIG': 'The Hartford',
        'CINF': 'Cincinnati Financial',
    }

    capital_data = []
    print("Initializing connection to financial APIs...")

    for ticker, name in tickers.items():
        try:
            company = yf.Ticker(ticker)
            balance_sheet = company.balance_sheet
            
            # Extract the most recent quarter's equity
            surplus = balance_sheet.loc['Stockholders Equity'].iloc[0]
            surplus_billions = surplus / 1_000_000_000
            
            capital_data.append({
                'Company': name,
                'Ticker': ticker,
                'Surplus_Billions': round(surplus_billions, 2)
            })
            print(f"[SUCCESS] Pulled capital for {name}")
            
        except Exception as e:
            print(f"[FAILED] Could not process {name}: {e}")

    # Convert to DataFrame
    df = pd.DataFrame(capital_data)
    
    # Save to CSV
    save_path = os.path.join('data', 'carrier_capital.csv')
    
    # Ensure the data folder exists
    os.makedirs('data', exist_ok=True)
    
    df.to_csv(save_path, index=False)
    print(f"\nData successfully saved to {save_path}")
    return df

if __name__ == "__main__":
    fetch_carrier_capital()