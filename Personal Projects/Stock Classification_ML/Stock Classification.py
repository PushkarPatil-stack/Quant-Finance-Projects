import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# ========================================
# HELPER FUNCTIONS
# ========================================

def get_stock_data_yfinance(ticker):
    """Fetch stock data from Yahoo Finance for new stocks"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        market_cap = info.get('marketCap', None)
        pe_ratio = info.get('trailingPE', None)
        pb_ratio = info.get('priceToBook', None)
        eps_growth = info.get('earningsGrowth', None)
        if eps_growth:
            eps_growth = eps_growth * 100
        return {
            'Ticker': ticker,
            'Name': info.get('longName', ticker),
            'Market_Cap': market_cap,
            'PE_Ratio': pe_ratio,
            'PB_Ratio': pb_ratio,
            'EPS_Growth': eps_growth
        }
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def assign_morningstar_label_g1(market_cap_millions, peg_ratio):
    """Assign Morningstar-style label based on industry standards"""
    market_cap_billions = market_cap_millions / 1000

    # Size categories (Morningstar approach)
    if market_cap_billions < 2:
        size = "Small"
    elif market_cap_billions < 10:
        size = "Mid"
    else:
        size = "Large"

    # Style categories based on PEG (simplified for PEG-based analysis)
    if peg_ratio < 1.0:
        style = "Value"
    elif peg_ratio <= 1.5:
        style = "Blend"
    else:
        style = "Growth"

    return f"{size}-cap {style}"

def assign_morningstar_label_g2(market_cap_millions, pb_ratio):
    """Assign Morningstar-style label based on industry standards"""
    market_cap_billions = market_cap_millions / 1000

    # Size categories
    if market_cap_billions < 2:
        size = "Small"
    elif market_cap_billions < 10:
        size = "Mid"
    else:
        size = "Large"

    # Style categories based on P/B (Morningstar uses P/B as part of value score)
    if pb_ratio < 2.0:
        style = "Value"
    elif pb_ratio <= 4.0:
        style = "Blend"
    else:
        style = "Growth"

    return f"{size}-cap {style}"

# ========================================
# PART 1: DATA LOADING & PREPARATION
# ========================================
print("=" * 80)
print("RUSSELL 2000 STOCK CLUSTERING ANALYSIS")
print("Unsupervised K-Means Clustering with Label Assignment")
print("=" * 80)

excel_file_path = r"C:\Users\Admin\Downloads\ML_Project.xlsx"
print(f"\nLoading data from: {excel_file_path}")

df_raw = pd.read_excel(excel_file_path)
print(f"\nExcel file loaded successfully!")
print(f"Total rows: {len(df_raw)}")

# Column mapping
COL_TICKER = 0
COL_PB_RATIO = 11
COL_EPS_GROWTH = 14
COL_MARKET_CAP = 15
COL_PE_RATIO = 16

# Extract relevant columns
df_raw['Ticker'] = df_raw.iloc[:, COL_TICKER]
df_raw['Market_Cap'] = pd.to_numeric(df_raw.iloc[:, COL_MARKET_CAP], errors='coerce')
df_raw['PE_Ratio'] = pd.to_numeric(df_raw.iloc[:, COL_PE_RATIO], errors='coerce')
df_raw['EPS_Growth'] = pd.to_numeric(df_raw.iloc[:, COL_EPS_GROWTH], errors='coerce')
df_raw['PB_Ratio'] = pd.to_numeric(df_raw.iloc[:, COL_PB_RATIO], errors='coerce')
df_raw = df_raw.replace('--', np.nan)

# Convert market cap from dollars to millions
df_raw['Market_Cap_Millions'] = df_raw['Market_Cap'] / 1_000_000

print("\n" + "=" * 80)
print("DATA PREPARATION & GROUPING")
print("=" * 80)
print(f"\nTotal companies in dataset: {len(df_raw)}")

# ========================================
# GROUP 1: Market Cap + PEG Ratio
# ========================================
print("\n" + "=" * 80)
print("GROUP 1 FILTERING: Market Cap + PEG Ratio")
print("=" * 80)

group1_mask = (
        df_raw['Market_Cap_Millions'].notna() &
        df_raw['PE_Ratio'].notna() &
        df_raw['EPS_Growth'].notna() &
        (df_raw['Market_Cap_Millions'] > 0) &
        (df_raw['PE_Ratio'] > 0) &
        (df_raw['EPS_Growth'] > 0)
)

df_group1_initial = df_raw[group1_mask].copy()
df_group1_initial['PEG_Ratio'] = df_group1_initial['PE_Ratio'] / df_group1_initial['EPS_Growth']

df_group1 = df_group1_initial[
    (df_group1_initial['PEG_Ratio'] > 0) &
    (df_group1_initial['PEG_Ratio'] < 10)
    ].copy()

print(f"\nGroup 1 Filtering Results:")
print(f"  - Final Group 1 companies: {len(df_group1)}")
print(f"\nGroup 1 Statistics:")
print(f"  Market Cap: ${df_group1['Market_Cap_Millions'].min():.0f}M to ${df_group1['Market_Cap_Millions'].max():.0f}M")
print(f"  PEG Ratio: {df_group1['PEG_Ratio'].min():.2f} to {df_group1['PEG_Ratio'].max():.2f}")

# ========================================
# GROUP 2: Market Cap + P/B Ratio
# ========================================
print("\n" + "=" * 80)
print("GROUP 2 FILTERING: Market Cap + P/B Ratio")
print("=" * 80)

group2_mask = (
        ~df_raw.index.isin(df_group1.index) &
        df_raw['Market_Cap_Millions'].notna() &
        df_raw['PB_Ratio'].notna() &
        (df_raw['Market_Cap_Millions'] > 0) &
        (df_raw['PB_Ratio'] > 0) &
        (df_raw['PB_Ratio'] < 20)
)

df_group2 = df_raw[group2_mask].copy()

print(f"\nGroup 2 Filtering Results:")
print(f"  - Final Group 2 companies: {len(df_group2)}")
print(f"\nGroup 2 Statistics:")
print(f"  Market Cap: ${df_group2['Market_Cap_Millions'].min():.0f}M to ${df_group2['Market_Cap_Millions'].max():.0f}M")
print(f"  P/B Ratio: {df_group2['PB_Ratio'].min():.2f} to {df_group2['PB_Ratio'].max():.2f}")

# ========================================
# GROUP 1: UNSUPERVISED K-MEANS CLUSTERING
# ========================================

if len(df_group1) >= 20:
    print("\n" + "=" * 80)
    print("GROUP 1: UNSUPERVISED K-MEANS CLUSTERING")
    print("=" * 80)

    X1 = df_group1[['Market_Cap_Millions', 'PEG_Ratio']].copy()
    X1['Market_Cap_Log'] = np.log10(X1['Market_Cap_Millions'])
    X1_features = X1[['Market_Cap_Log', 'PEG_Ratio']]

    scaler1 = StandardScaler()
    X1_scaled = scaler1.fit_transform(X1_features)

    print("\nRunning Elbow Method...")
    k_range = range(2, min(16, len(df_group1)//5))
    inertias1 = []
    silhouette_scores1 = []

    for k in k_range:
        kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=50, max_iter=300)
        kmeans_test.fit(X1_scaled)
        inertias1.append(kmeans_test.inertia_)
        silhouette_scores1.append(silhouette_score(X1_scaled, kmeans_test.labels_))
        print(f"k={k}: Inertia={kmeans_test.inertia_:.2f}, Silhouette={silhouette_scores1[-1]:.4f}")

    print(f"\nSelected k=9 for analysis (as per assignment)")

    n_clusters1 = 9
    kmeans1 = KMeans(n_clusters=n_clusters1, random_state=42, n_init=100, max_iter=500)
    df_group1['Cluster'] = kmeans1.fit_predict(X1_scaled)

    silhouette_avg1 = silhouette_score(X1_scaled, df_group1['Cluster'])
    print(f"Silhouette Score: {silhouette_avg1:.4f}")
    print(f"Inertia: {kmeans1.inertia_:.4f}")

    cluster_centers1 = scaler1.inverse_transform(kmeans1.cluster_centers_)

    # ========================================
    # STEP 1: SHOW PURE CLUSTER CHARACTERISTICS
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 1: K-MEANS CLUSTER CHARACTERISTICS (Unsupervised)")
    print("=" * 80)

    cluster_stats_g1 = []
    for i in range(n_clusters1):
        cluster_data = df_group1[df_group1['Cluster'] == i]
        count = len(cluster_data)
        market_cap_centroid = 10 ** cluster_centers1[i, 0]
        peg_centroid = cluster_centers1[i, 1]

        cluster_stats_g1.append({
            'Cluster': i,
            'Count': count,
            'Percentage': f"{count/len(df_group1)*100:.1f}%",
            'Centroid_MarketCap_M': market_cap_centroid,
            'Centroid_MarketCap_B': market_cap_centroid/1000,
            'Centroid_PEG': peg_centroid
        })

        print(f"\nCluster {i}:")
        print(f"  Stocks: {count} ({count/len(df_group1)*100:.1f}%)")
        print(f"  Centroid Market Cap: ${market_cap_centroid:.2f}M (${market_cap_centroid/1000:.2f}B)")
        print(f"  Centroid PEG: {peg_centroid:.2f}")

    # ========================================
    # STEP 2: ASSIGN LABELS - K-MEANS APPROACH
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 2: LABEL ASSIGNMENT - K-MEANS APPROACH")
    print("Labels based on relative positioning within discovered clusters")
    print("=" * 80)

    # Sort clusters by market cap to assign relative size labels
    sorted_by_mc = sorted(cluster_stats_g1, key=lambda x: x['Centroid_MarketCap_M'])

    # Sort by PEG to assign relative style labels
    sorted_by_peg = sorted(cluster_stats_g1, key=lambda x: x['Centroid_PEG'])

    kmeans_labels_g1 = {}

    for i in range(n_clusters1):
        mc_rank = next(idx for idx, c in enumerate(sorted_by_mc) if c['Cluster'] == i)
        peg_rank = next(idx for idx, c in enumerate(sorted_by_peg) if c['Cluster'] == i)

        # Size based on relative position (tertiles)
        if mc_rank < 3:
            size = "Small"
        elif mc_rank < 6:
            size = "Mid"
        else:
            size = "Large"

        # Style based on relative position (tertiles)
        if peg_rank < 3:
            style = "Value"
        elif peg_rank < 6:
            style = "Blend"
        else:
            style = "Growth"

        kmeans_labels_g1[i] = f"{size} {style}"

    df_group1['Label_KMeans'] = df_group1['Cluster'].map(kmeans_labels_g1)

    print("\nK-Means Based Labels (Relative Positioning):")
    for i in range(n_clusters1):
        print(f"  Cluster {i}: {kmeans_labels_g1[i]}")

    # ========================================
    # STEP 3: ASSIGN LABELS - USER DEFINED EXPECTATIONS
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 3: LABEL ASSIGNMENT - USER DEFINED EXPECTATIONS")
    print("=" * 80)

    morningstar_labels_g1 = {}

    for i in range(n_clusters1):
        market_cap_centroid = 10 ** cluster_centers1[i, 0]
        peg_centroid = cluster_centers1[i, 1]
        morningstar_labels_g1[i] = assign_morningstar_label_g1(market_cap_centroid, peg_centroid)

    df_group1['User_Defined_Label'] = df_group1['Cluster'].map(morningstar_labels_g1)


    print("\nUser Defined Labels:")
    print("(Size: <$2B=Small, $2-10B=Mid, >$10B=Large)")
    print("(Style: PEG<1.0=Value, 1.0-1.5=Blend, >1.5=Growth)")
    for i in range(n_clusters1):
        print(f"  Cluster {i}: {morningstar_labels_g1[i]}")

    # ========================================
    # STEP 4: COMPARE K-MEANS vs MORNINGSTAR
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 4: COMPARISON - K-MEANS vs USER DEFINED LABELING")
    print("=" * 80)

    print(f"\n{'Cluster':<10} {'K-Means Label':<20} {'User Defined Label':<20} {'Match?':<10}")
    print("-" * 70)

    for i in range(n_clusters1):
        kmeans_label = kmeans_labels_g1[i]
        morningstar_label = morningstar_labels_g1[i]
        match = "✓" if kmeans_label == morningstar_label else "✗"
        print(f"{i:<10} {kmeans_label:<20} {morningstar_label:<20} {match:<10}")

    # Count agreements
    agreements = sum(1 for i in range(n_clusters1) if kmeans_labels_g1[i] == morningstar_labels_g1[i])
    print(f"\nAgreement: {agreements}/{n_clusters1} clusters ({agreements/n_clusters1*100:.1f}%)")

# ========================================
# GROUP 2: UNSUPERVISED K-MEANS CLUSTERING
# ========================================

if len(df_group2) >= 20:
    print("\n" + "=" * 80)
    print("GROUP 2: UNSUPERVISED K-MEANS CLUSTERING")
    print("=" * 80)

    X2 = df_group2[['Market_Cap_Millions', 'PB_Ratio']].copy()
    X2['Market_Cap_Log'] = np.log10(X2['Market_Cap_Millions'])
    X2_features = X2[['Market_Cap_Log', 'PB_Ratio']]

    scaler2 = StandardScaler()
    X2_scaled = scaler2.fit_transform(X2_features)

    print("\nRunning Elbow Method...")
    k_range = range(2, min(16, len(df_group2)//5))
    inertias2 = []
    silhouette_scores2 = []

    for k in k_range:
        kmeans_test = KMeans(n_clusters=k, random_state=42, n_init=50, max_iter=300)
        kmeans_test.fit(X2_scaled)
        inertias2.append(kmeans_test.inertia_)
        silhouette_scores2.append(silhouette_score(X2_scaled, kmeans_test.labels_))
        print(f"k={k}: Inertia={kmeans_test.inertia_:.2f}, Silhouette={silhouette_scores2[-1]:.4f}")

    print(f"\nSelected k=9 for analysis (as per assignment)")

    n_clusters2 = 9
    kmeans2 = KMeans(n_clusters=n_clusters2, random_state=42, n_init=100, max_iter=500)
    df_group2['Cluster'] = kmeans2.fit_predict(X2_scaled)

    silhouette_avg2 = silhouette_score(X2_scaled, df_group2['Cluster'])
    print(f"Silhouette Score: {silhouette_avg2:.4f}")
    print(f"Inertia: {kmeans2.inertia_:.4f}")

    cluster_centers2 = scaler2.inverse_transform(kmeans2.cluster_centers_)

    # ========================================
    # STEP 1: SHOW PURE CLUSTER CHARACTERISTICS
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 1: K-MEANS CLUSTER CHARACTERISTICS (Unsupervised)")
    print("=" * 80)

    cluster_stats_g2 = []
    for i in range(n_clusters2):
        cluster_data = df_group2[df_group2['Cluster'] == i]
        count = len(cluster_data)
        market_cap_centroid = 10 ** cluster_centers2[i, 0]
        pb_centroid = cluster_centers2[i, 1]

        cluster_stats_g2.append({
            'Cluster': i,
            'Count': count,
            'Percentage': f"{count/len(df_group2)*100:.1f}%",
            'Centroid_MarketCap_M': market_cap_centroid,
            'Centroid_MarketCap_B': market_cap_centroid/1000,
            'Centroid_PB': pb_centroid
        })

        print(f"\nCluster {i}:")
        print(f"  Stocks: {count} ({count/len(df_group2)*100:.1f}%)")
        print(f"  Centroid Market Cap: ${market_cap_centroid:.2f}M (${market_cap_centroid/1000:.2f}B)")
        print(f"  Centroid P/B: {pb_centroid:.2f}")

    # ========================================
    # STEP 2: ASSIGN LABELS - K-MEANS APPROACH
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 2: LABEL ASSIGNMENT - K-MEANS APPROACH")
    print("=" * 80)

    sorted_by_mc = sorted(cluster_stats_g2, key=lambda x: x['Centroid_MarketCap_M'])
    sorted_by_pb = sorted(cluster_stats_g2, key=lambda x: x['Centroid_PB'])

    kmeans_labels_g2 = {}

    for i in range(n_clusters2):
        mc_rank = next(idx for idx, c in enumerate(sorted_by_mc) if c['Cluster'] == i)
        pb_rank = next(idx for idx, c in enumerate(sorted_by_pb) if c['Cluster'] == i)

        if mc_rank < 3:
            size = "Small"
        elif mc_rank < 6:
            size = "Mid"
        else:
            size = "Large"

        if pb_rank < 3:
            style = "Value"
        elif pb_rank < 6:
            style = "Blend"
        else:
            style = "Growth"

        kmeans_labels_g2[i] = f"{size} {style}"

    df_group2['Label_KMeans'] = df_group2['Cluster'].map(kmeans_labels_g2)

    print("\nK-Means Based Labels:")
    for i in range(n_clusters2):
        print(f"  Cluster {i}: {kmeans_labels_g2[i]}")

    # ========================================
    # STEP 3: ASSIGN LABELS - USER DEFINED EXPECTATIONS
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 3: LABEL ASSIGNMENT - USER DEFINED EXPECTATIONS")
    print("=" * 80)

    morningstar_labels_g2 = {}

    for i in range(n_clusters2):
        market_cap_centroid = 10 ** cluster_centers2[i, 0]
        pb_centroid = cluster_centers2[i, 1]
        morningstar_labels_g2[i] = assign_morningstar_label_g2(market_cap_centroid, pb_centroid)

    df_group2['User_Defined_Label'] = df_group2['Cluster'].map(morningstar_labels_g2)

    print("\nUser Defined Labels:")
    print("(Size: <$2B=Small, $2-10B=Mid, >$10B=Large)")
    print("(Style: P/B<2.0=Value, 2.0-4.0=Blend, >4.0=Growth)")
    for i in range(n_clusters2):
        print(f"  Cluster {i}: {morningstar_labels_g2[i]}")

    # ========================================
    # STEP 4: COMPARE K-MEANS vs USER DEFINED
    # ========================================
    print("\n" + "=" * 80)
    print("STEP 4: COMPARISON - K-MEANS vs USER DEFINED LABELING")
    print("=" * 80)

    print(f"\n{'Cluster':<10} {'K-Means Label':<20} {'User Defined Label':<20} {'Match?':<10}")
    print("-" * 70)

    for i in range(n_clusters2):
        kmeans_label = kmeans_labels_g2[i]
        morningstar_label = morningstar_labels_g2[i]
        match = "✓" if kmeans_label == morningstar_label else "✗"
        print(f"{i:<10} {kmeans_label:<20} {morningstar_label:<20} {match:<10}")

    agreements = sum(1 for i in range(n_clusters2) if kmeans_labels_g2[i] == morningstar_labels_g2[i])
    print(f"\nAgreement: {agreements}/{n_clusters2} clusters ({agreements/n_clusters2*100:.1f}%)")

# ========================================
# SAMPLE STOCKS FOR MANUAL COMPARISON
# ========================================
print("\n" + "=" * 80)
print("SAMPLE STOCKS - FOR MANUAL COMPARISON")
print("=" * 80)

if len(df_group1) >= 5:
    print("\n" + "="*80)
    print("GROUP 1 SAMPLES:")
    print("="*80)
    sample_tickers_group1 = ['VLY', 'HBCP', 'PKBK', 'NRIM', 'FLR']

    print(f"\n{'Ticker':<8} {'Market Cap':<15} {'PEG':<8} {'Cluster':<10} {'K-Means Label':<20} {'User Defined Label':<20}")
    print("-" * 110)

    for ticker in sample_tickers_group1:
        if ticker in df_group1['Ticker'].values:
            row = df_group1[df_group1['Ticker'] == ticker].iloc[0]
            mc_b = row['Market_Cap_Millions']/1000
            print(f"{ticker:<8} ${mc_b:<6.2f}B{'':<6} {row['PEG_Ratio']:<8.2f} {row['Cluster']:<10} {row['Label_KMeans']:<20} {row['Label_Morningstar']:<20}")

if len(df_group2) >= 5:
    print("\n" + "="*80)
    print("GROUP 2 SAMPLES:")
    print("="*80)
    sample_tickers_group2 = ['TG', 'AFRI', 'ULH', 'FFIN', 'IDT']

    print(f"\n{'Ticker':<8} {'Market Cap':<15} {'P/B':<8} {'Cluster':<10} {'K-Means Label':<20} {'User Defined Label':<20}")
    print("-" * 110)

    for ticker in sample_tickers_group2:
        if ticker in df_group2['Ticker'].values:
            row = df_group2[df_group2['Ticker'] == ticker].iloc[0]
            mc_b = row['Market_Cap_Millions']/1000
            print(f"{ticker:<8} ${mc_b:<6.2f}B{'':<6} {row['PB_Ratio']:<8.2f} {row['Cluster']:<10} {row['Label_KMeans']:<20} {row['Label_Morningstar']:<20}")

# ========================================
# KNN CLASSIFICATION FOR NEW STOCKS
# ========================================
print("\n" + "=" * 80)
print("KNN CLASSIFICATION FOR NEW STOCKS")
print("=" * 80)

if len(df_group1) >= 20:
    knn1 = KNeighborsClassifier(n_neighbors=5)
    knn1.fit(X1_scaled, df_group1['Cluster'])
    knn_accuracy1 = knn1.score(X1_scaled, df_group1['Cluster'])
    print(f"\nGroup 1 KNN Training Accuracy: {knn_accuracy1:.4f}")

if len(df_group2) >= 20:
    knn2 = KNeighborsClassifier(n_neighbors=5)
    knn2.fit(X2_scaled, df_group2['Cluster'])
    knn_accuracy2 = knn2.score(X2_scaled, df_group2['Cluster'])
    print(f"Group 2 KNN Training Accuracy: {knn_accuracy2:.4f}")

new_tickers = ['DOCN', 'BKKT']
print("\n" + "="*80)
print(f"Classifying New Stocks: {', '.join(new_tickers)}")
print("="*80)

for ticker in new_tickers:
    print(f"\n{ticker}:")
    stock_data = get_stock_data_yfinance(ticker)
    if stock_data is None:
        continue

    market_cap_millions = stock_data['Market_Cap'] / 1e6 if stock_data['Market_Cap'] else None
    print(f"  Market Cap: ${market_cap_millions:,.0f}M (${stock_data['Market_Cap']/1e9:.2f}B)")
    print(f"  PE: {stock_data['PE_Ratio']}, P/B: {stock_data['PB_Ratio']}, EPS Growth: {stock_data['EPS_Growth']}")

    can_use_group1 = (market_cap_millions and stock_data['PE_Ratio'] and
                      stock_data['EPS_Growth'] and stock_data['EPS_Growth'] > 0)

    if can_use_group1:
        peg_ratio = stock_data['PE_Ratio'] / stock_data['EPS_Growth']
        if 0 < peg_ratio < 10:
            print(f"  PEG Ratio: {peg_ratio:.2f}")
            market_cap_log = np.log10(market_cap_millions)
            X_test = scaler1.transform([[market_cap_log, peg_ratio]])
            predicted_cluster = knn1.predict(X_test)[0]
            kmeans_label = kmeans_labels_g1[predicted_cluster]
            morningstar_label = morningstar_labels_g1[predicted_cluster]
            print(f"  → GROUP 1 Classification:")
            print(f"     Cluster: {predicted_cluster}")
            print(f"     K-Means Label: {kmeans_label}")
            print(f"     User Defined Label: {morningstar_label}")
            continue

    if market_cap_millions and stock_data['PB_Ratio'] and 0 < stock_data['PB_Ratio'] < 20:
        print(f"  P/B Ratio: {stock_data['PB_Ratio']:.2f}")
        market_cap_log = np.log10(market_cap_millions)
        X_test = scaler2.transform([[market_cap_log, stock_data['PB_Ratio']]])
        predicted_cluster = knn2.predict(X_test)[0]
        kmeans_label = kmeans_labels_g2[predicted_cluster]
        morningstar_label = morningstar_labels_g2[predicted_cluster]
        print(f"  → GROUP 2 Classification:")
        print(f"     Cluster: {predicted_cluster}")
        print(f"     K-Means Label: {kmeans_label}")
        print(f"     User Defined Label: {morningstar_label}")

# ========================================
# VISUALIZATIONS
# ========================================
print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)

if len(df_group1) >= 20:
    # Group 1: Cluster Visualization
    fig1 = plt.figure(figsize=(20, 8))

    # Left: Data distribution with user defined boundaries
    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(df_group1['Market_Cap_Millions'], df_group1['PEG_Ratio'],
                alpha=0.6, s=80, c='steelblue', edgecolors='black', linewidth=0.5)
    ax1.set_xscale('log')
    ax1.set_xlabel('Market Cap (Millions $)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('PEG Ratio', fontsize=13, fontweight='bold')
    ax1.set_title('Group 1: Data Distribution\nWith User Defined Boundaries', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # User defined boundaries
    ax1.axvline(x=2000, color='red', linestyle=':', linewidth=2.5, alpha=0.7, label='Small/Mid ($2B)')
    ax1.axvline(x=10000, color='orange', linestyle=':', linewidth=2.5, alpha=0.7, label='Mid/Large ($10B)')
    ax1.axhline(y=1.0, color='green', linestyle=':', linewidth=2.5, alpha=0.7, label='Value/Blend (PEG=1.0)')
    ax1.axhline(y=1.5, color='purple', linestyle=':', linewidth=2.5, alpha=0.7, label='Blend/Growth (PEG=1.5)')
    ax1.legend(loc='upper right', fontsize=9)

    stats_text = f'n = {len(df_group1)}\nUser Defined Boundaries:\n• Size: $2B, $10B\n• Style: PEG 1.0, 1.5'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Right: K-Means discovered clusters
    ax2 = plt.subplot(1, 2, 2)
    scatter = ax2.scatter(df_group1['Market_Cap_Millions'], df_group1['PEG_Ratio'],
                          c=df_group1['Cluster'], cmap='tab10',
                          alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
    ax2.set_xscale('log')
    ax2.set_xlabel('Market Cap (Millions $)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('PEG Ratio', fontsize=13, fontweight='bold')
    ax2.set_title('Group 1: K-Means Discovered Clusters (k=9)\nUnsupervised Learning Results', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    centers1_original = scaler1.inverse_transform(kmeans1.cluster_centers_)
    centers1_market_cap = 10 ** centers1_original[:, 0]
    ax2.scatter(centers1_market_cap, centers1_original[:, 1], marker='X', s=500, c='red',
                edgecolors='black', linewidths=3, label='Centroids', zorder=10)
    for i, (mc, peg) in enumerate(zip(centers1_market_cap, centers1_original[:, 1])):
        ax2.annotate(str(i), (mc, peg), fontsize=14, fontweight='bold', ha='center', va='center',
                     color='white', bbox=dict(boxstyle='circle,pad=0.3', facecolor='red',
                                              edgecolor='black', linewidth=2.5))
    plt.colorbar(scatter, ax=ax2, label='Cluster ID')
    ax2.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.savefig('1_group1_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: 1_group1_clusters.png")

if len(df_group1) >= 20:
    # Group 1: Elbow and Silhouette
    fig2 = plt.figure(figsize=(18, 7))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(k_range, inertias1, 'bo-', linewidth=3, markersize=10, label='Inertia')
    ax1.axvline(x=9, color='red', linestyle='--', linewidth=2.5, alpha=0.8, label='k=9 (selected)')
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inertia (WCSS)', fontsize=12, fontweight='bold')
    ax1.set_title('Group 1: Elbow Method', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(k_range, silhouette_scores1, 'go-', linewidth=3, markersize=10, label='Silhouette')
    ax2.axvline(x=9, color='red', linestyle='--', linewidth=2.5, alpha=0.8, label='k=9 (selected)')
    ax2.axhline(y=0.5, color='orange', linestyle=':', linewidth=2, label='Good threshold (0.5)')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax2.set_title('Group 1: Silhouette Analysis', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('2_group1_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: 2_group1_metrics.png")

if len(df_group2) >= 20:
    # Group 2: Cluster Visualization
    fig3 = plt.figure(figsize=(20, 8))

    ax1 = plt.subplot(1, 2, 1)
    ax1.scatter(df_group2['Market_Cap_Millions'], df_group2['PB_Ratio'],
                alpha=0.6, s=80, c='coral', edgecolors='black', linewidth=0.5)
    ax1.set_xscale('log')
    ax1.set_xlabel('Market Cap (Millions $)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('P/B Ratio', fontsize=13, fontweight='bold')
    ax1.set_title('Group 2: Data Distribution\nWith User Defined Boundaries', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # User defined boundaries
    ax1.axvline(x=2000, color='red', linestyle=':', linewidth=2.5, alpha=0.7, label='Small/Mid ($2B)')
    ax1.axvline(x=10000, color='orange', linestyle=':', linewidth=2.5, alpha=0.7, label='Mid/Large ($10B)')
    ax1.axhline(y=2.0, color='green', linestyle=':', linewidth=2.5, alpha=0.7, label='Value/Blend (P/B=2.0)')
    ax1.axhline(y=4.0, color='purple', linestyle=':', linewidth=2.5, alpha=0.7, label='Blend/Growth (P/B=4.0)')
    ax1.legend(loc='upper right', fontsize=9)

    stats_text = f'n = {len(df_group2)}\nUser Defined Boundaries:\n• Size: $2B, $10B\n• Style: P/B 2.0, 4.0'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax2 = plt.subplot(1, 2, 2)
    scatter = ax2.scatter(df_group2['Market_Cap_Millions'], df_group2['PB_Ratio'],
                          c=df_group2['Cluster'], cmap='tab10',
                          alpha=0.7, s=80, edgecolors='black', linewidth=0.5)
    ax2.set_xscale('log')
    ax2.set_xlabel('Market Cap (Millions $)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('P/B Ratio', fontsize=13, fontweight='bold')
    ax2.set_title('Group 2: K-Means Discovered Clusters (k=9)\nUnsupervised Learning Results', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')

    centers2_original = scaler2.inverse_transform(kmeans2.cluster_centers_)
    centers2_market_cap = 10 ** centers2_original[:, 0]
    ax2.scatter(centers2_market_cap, centers2_original[:, 1], marker='X', s=500, c='red',
                edgecolors='black', linewidths=3, label='Centroids', zorder=10)
    for i, (mc, pb) in enumerate(zip(centers2_market_cap, centers2_original[:, 1])):
        ax2.annotate(str(i), (mc, pb), fontsize=14, fontweight='bold', ha='center', va='center',
                     color='white', bbox=dict(boxstyle='circle,pad=0.3', facecolor='red',
                                              edgecolor='black', linewidth=2.5))
    plt.colorbar(scatter, ax=ax2, label='Cluster ID')
    ax2.legend(loc='upper right', fontsize=11)
    plt.tight_layout()
    plt.savefig('3_group2_clusters.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: 3_group2_clusters.png")

if len(df_group2) >= 20:
    # Group 2: Elbow and Silhouette
    fig4 = plt.figure(figsize=(18, 7))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(k_range, inertias2, 'bo-', linewidth=3, markersize=10, label='Inertia')
    ax1.axvline(x=9, color='red', linestyle='--', linewidth=2.5, alpha=0.8, label='k=9 (selected)')
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inertia (WCSS)', fontsize=12, fontweight='bold')
    ax1.set_title('Group 2: Elbow Method', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(k_range, silhouette_scores2, 'go-', linewidth=3, markersize=10, label='Silhouette')
    ax2.axvline(x=9, color='red', linestyle='--', linewidth=2.5, alpha=0.8, label='k=9 (selected)')
    ax2.axhline(y=0.5, color='orange', linestyle=':', linewidth=2, label='Good threshold (0.5)')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    ax2.set_title('Group 2: Silhouette Analysis', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('4_group2_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✓ Saved: 4_group2_metrics.png")

print("\n" + "=" * 80)
print("All visualizations generated successfully!")
print("=" * 80)

# ========================================
# SAVE RESULTS
# ========================================
if len(df_group1) >= 20:
    df_group1.to_excel('5_group1_analysis.xlsx', index=False)
    print("\n✓ Saved: 5_group1_analysis.xlsx")

if len(df_group2) >= 20:
    df_group2.to_excel('6_group2_analysis.xlsx', index=False)
    print("✓ Saved: 6_group2_analysis.xlsx")

# ========================================
# FINAL SUMMARY
# ========================================
print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)

if len(df_group1) >= 20:
    print("\n" + "="*80)
    print("GROUP 1 RESULTS:")
    print("="*80)
    print(f"  Stocks Analyzed: {len(df_group1)}")
    print(f"  Clusters: 9")
    print(f"  Silhouette Score: {silhouette_avg1:.4f}")
    print(f"  KNN Accuracy: {knn_accuracy1:.4f}")

    agreements = sum(1 for i in range(n_clusters1) if kmeans_labels_g1[i] == morningstar_labels_g1[i])
    print(f"  Label Agreement (K-Means vs User Defined): {agreements}/9 ({agreements/9*100:.1f}%)")

if len(df_group2) >= 20:
    print("\n" + "="*80)
    print("GROUP 2 RESULTS:")
    print("="*80)
    print(f"  Stocks Analyzed: {len(df_group2)}")
    print(f"  Clusters: 9")
    print(f"  Silhouette Score: {silhouette_avg2:.4f}")
    print(f"  KNN Accuracy: {knn_accuracy2:.4f}")

    agreements = sum(1 for i in range(n_clusters2) if kmeans_labels_g2[i] == morningstar_labels_g2[i])
    print(f"  Label Agreement (K-Means vs User Defined): {agreements}/9 ({agreements/9*100:.1f}%)")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)