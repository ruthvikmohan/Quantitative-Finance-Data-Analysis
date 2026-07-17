###Python Code: Yield Curve Modeling

##Required Libraries

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline


##Input Data: Indian Government Bond Yields

# Maturities in years and corresponding yields in %
maturities = np.array([0.5, 1, 2, 5, 10, 20, 30])
yields = np.array([5.43, 5.47, 5.68, 6.04, 6.29, 6.81, 7.01])

##Nelson-Siegel Model

# Nelson-Siegel yield function
def nelson_siegel(tau, beta0, beta1, beta2, lambd):
    term1 = (1 - np.exp(-lambd * tau)) / (lambd * tau)
    term2 = term1 - np.exp(-lambd * tau)
    return beta0 + beta1 * term1 + beta2 * term2

# Objective function to minimize
def ns_objective(params, tau, y_obs):
    beta0, beta1, beta2, lambd = params
    y_fit = nelson_siegel(tau, beta0, beta1, beta2, lambd)
    return np.sum((y_obs - y_fit) ** 2)

# Initial guess
initial_params = [6.0, -1.0, 1.0, 0.5]

# Fit model
result = minimize(ns_objective, initial_params, args=(maturities, yields), method='L-BFGS-B')
beta0, beta1, beta2, lambd = result.x
print(f"Nelson-Siegel Parameters:\nβ0 = {beta0:.4f}, β1 = {beta1:.4f}, β2 = {beta2:.4f}, λ = {lambd:.4f}")

##Cubic Spline Model

# Fit cubic spline
cs = CubicSpline(maturities, yields)


##Plot Both Models

# Plotting
tau_plot = np.linspace(0.5, 30, 300)
ns_yields = nelson_siegel(tau_plot, beta0, beta1, beta2, lambd)
cs_yields = cs(tau_plot)

plt.figure(figsize=(10, 6))
plt.plot(maturities, yields, 'o', label='Observed Yields', color='black')
plt.plot(tau_plot, ns_yields, label='Nelson-Siegel Fit', color='blue')
plt.plot(tau_plot, cs_yields, label='Cubic Spline Fit', color='green', linestyle='--')
plt.title('Yield Curve Fitting: Nelson-Siegel vs Cubic Spline')
plt.xlabel('Maturity (Years)')
plt.ylabel('Yield (%)')
plt.legend()
plt.grid(True)
plt.show()


###Empirical Analysis of ETF's

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup

# Step a: Get the top 30 holdings of XLK
def get_top_holdings():
    url = "https://finance.yahoo.com/quote/XLK/holdings/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        holding_section = soup.find('section', {'data-test': 'holdings'})
        symbols = []
        for li in holding_section.find_all('li', {'data-test': 'holding-list-item'}):
            symbol = li.find('span').text.strip()
            symbols.append(symbol)
        return symbols[:30]  # Top 30 holdings
    except Exception as e:
        print(f"Error retrieving holdings: {e}")
        # Fallback to static list if scraping fails
        return ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ADBE', 'CSCO', 'ORCL', 'ACN', 'TXN', 'CRM', 
                'AMD', 'INTU', 'IBM', 'QCOM', 'FIS', 'FISV', 'AMAT', 'MSI', 'MU', 'NOW', 
                'APH', 'LRCX', 'ADI', 'TEL', 'KLAC', 'SNPS', 'CDNS', 'MCHP', 'FTNT', 'CTSH']

# Retrieve holdings and handle any missing symbols
holdings = get_top_holdings()
print("Top 30 Holdings:", holdings)

# Step b: Get at least 6 months of historical data (~120 points)
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=190)).strftime('%Y-%m-%d')  # ~6.3 months
data = yf.download(holdings, start=start_date, end=end_date)['Adj Close']
data = data.dropna(axis=1)  # Remove stocks with missing data
print(f"\nData shape after download: {data.shape}")

# Step c: Compute daily returns
returns = data.pct_change().dropna()
print(f"\nReturns DataFrame shape: {returns.shape}")

# Step d: Compute covariance matrix
cov_matrix = returns.cov()
print("\nCovariance Matrix (first 5x5):")
print(cov_matrix.iloc[:5, :5].to_string())

# Step e: Compute PCA via covariance matrix eigendecomposition
centered_returns = returns - returns.mean()
cov_matrix_pca = centered_returns.cov()
eigenvals_pca, eigenvecs_pca = np.linalg.eigh(cov_matrix_pca)
# Sort eigenvalues in descending order
idx = eigenvals_pca.argsort()[::-1]
eigenvals_pca = eigenvals_pca[idx]
eigenvecs_pca = eigenvecs_pca[:, idx]
print("\nPCA Eigenvalues (first 5):")
print(eigenvals_pca[:5])
print("\nPCA Eigenvectors (first 5 components):")
print(eigenvecs_pca[:, :5])

# Step f: Compute SVD of the centered returns matrix
U, s, Vt = np.linalg.svd(centered_returns, full_matrices=False)
print("\nSVD Results:")
print("U shape:", U.shape, "Singular values:", s, "Vt shape:", Vt.shape)
print("\nFirst 5 singular values:")
print(s[:5])
print("\nVt matrix (first 5 rows):")
print(Vt[:5, :])