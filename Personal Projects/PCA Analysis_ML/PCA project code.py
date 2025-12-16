import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Define tickers (S&P 500 names and symbols)
# Palantir [finance:Palantir Technologies Inc.]
# Microsoft [finance:Microsoft Corporation]
# Nvidia [finance:NVIDIA Corporation]
# Amazon [finance:Amazon.com, Inc.]
# Apple [finance:Apple Inc.]
# Netflix [finance:Netflix, Inc.]
# Meta [finance:Meta Platforms, Inc.]
# Alphabet [finance:Alphabet Inc.]
# Taiwan Semiconductor [finance:Taiwan Semiconductor Manufacturing Company Limited]
# Tesla [finance:Tesla, Inc.]
# Oracle [finance:Oracle Corporation]
# AMD [finance:Advanced Micro Devices, Inc.]
# IBM [finance:International Business Machines Corporation]

tickers = {
    "Palantir": "PLTR",
    "Microsoft": "MSFT",
    "Nvidia": "NVDA",
    "Amazon": "AMZN",
    "Apple": "AAPL",
    "Netflix": "NFLX",
    "Meta": "META",
    "Alphabet": "GOOGL",
    "Taiwan Semiconductor": "TSM",
    "Tesla": "TSLA",
    "Oracle": "ORCL",
    "AMD": "AMD",
    "IBM": "IBM",
}

start_date = "2023-11-27"
end_date = "2025-11-27"

# ---------------- Data -----------------
raw_data = yf.download(list(tickers.values()),
                       start=start_date, end=end_date,
                       auto_adjust=False)

prices = raw_data["Adj Close"].dropna()

# log returns
returns = np.log(prices / prices.shift(1)).dropna()

# centered data for covariance-PCA
X = returns - returns.mean()        # (T x N)
X_mat = X.values

# quick sanity check
print("Returns shape:", returns.shape)
print("Returns variance:\n", returns.var())

# =====================================================
# 1) PCA on COVARIANCE matrix
# =====================================================
Sigma_cov = np.cov(X_mat.T)         # covariance matrix (N x N)

# show covariance matrix in the output
cov_df = pd.DataFrame(Sigma_cov,
                      index=returns.columns,
                      columns=returns.columns)
print("Covariance matrix:\n", cov_df)
# save to CSV
cov_df.to_csv("cov_matrix.csv")

# eigen-decomposition (matrix is symmetric, so eigh is safe)
eig_vals_cov, eig_vecs_cov = np.linalg.eigh(Sigma_cov)

# sort descending
idx_cov = eig_vals_cov.argsort()[::-1]
eig_vals_cov = eig_vals_cov[idx_cov]
eig_vecs_cov = eig_vecs_cov[:, idx_cov]

# checks
print("Cov trace:", np.trace(Sigma_cov))
print("Cov eigenvalue sum:", eig_vals_cov.sum())

# explained variance (cov)
var_ratio_cov = eig_vals_cov / eig_vals_cov.sum()
print("Cov: first 3 ratios:", var_ratio_cov[:3],
      " total=", var_ratio_cov[:3].sum())

# scores (PC time series) for covariance PCA
V_cov = eig_vecs_cov[:, :3]         # (N x 3)
scores_cov = X_mat @ V_cov          # (T x 3)

scores_cov = pd.DataFrame(scores_cov,
                          index=X.index,
                          columns=["PC1_cov", "PC2_cov", "PC3_cov"])

# plot covariance PCs
plt.figure(figsize=(10, 6))
plt.plot(scores_cov.index, scores_cov["PC1_cov"], label="PC1 cov")
plt.plot(scores_cov.index, scores_cov["PC2_cov"], label="PC2 cov")
plt.plot(scores_cov.index, scores_cov["PC3_cov"], label="PC3 cov")
plt.xlabel("Date")
plt.ylabel("Principal Component Score")
plt.title("First Three PCs (Covariance Matrix)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# loadings table
loadings_cov = pd.DataFrame(V_cov,
                            index=returns.columns,
                            columns=["PC1_cov", "PC2_cov", "PC3_cov"])
print("Covariance PC loadings:\n", loadings_cov)

# =====================================================
# 2) PCA on CORRELATION matrix
# =====================================================
# standardize returns for correlation PCA
Z = (returns - returns.mean()) / returns.std(ddof=0)  # (T x N)
Z_mat = Z.values

R = np.corrcoef(Z_mat.T)          # correlation matrix (N x N)

eig_vals_cor, eig_vecs_cor = np.linalg.eigh(R)

# sort descending
idx_cor = eig_vals_cor.argsort()[::-1]
eig_vals_cor = eig_vals_cor[idx_cor]
eig_vecs_cor = eig_vecs_cor[:, idx_cor]

print("Cor trace:", np.trace(R))
print("Cor eigenvalue sum:", eig_vals_cor.sum())

var_ratio_cor = eig_vals_cor / eig_vals_cor.sum()
print("Cor: first 3 ratios:", var_ratio_cor[:3],
      " total=", var_ratio_cor[:3].sum())

# scores for correlation PCA
V_cor = eig_vecs_cor[:, :3]        # (N x 3)
scores_cor = Z_mat @ V_cor         # (T x 3)

scores_cor = pd.DataFrame(scores_cor,
                          index=Z.index,
                          columns=["PC1_cor", "PC2_cor", "PC3_cor"])

plt.figure(figsize=(10, 6))
plt.plot(scores_cor.index, scores_cor["PC1_cor"], label="PC1 cor")
plt.plot(scores_cor.index, scores_cor["PC2_cor"], label="PC2 cor")
plt.plot(scores_cor.index, scores_cor["PC3_cor"], label="PC3 cor")
plt.xlabel("Date")
plt.ylabel("Principal Component Score")
plt.title("First Three PCs (Correlation Matrix)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

loadings_cor = pd.DataFrame(V_cor,
                            index=returns.columns,
                            columns=["PC1_cor", "PC2_cor", "PC3_cor"])
print("Correlation PC loadings:\n", loadings_cor)
