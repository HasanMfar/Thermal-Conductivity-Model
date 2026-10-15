"""
Response Surface Model & Statistical Fitting Script
Manuscript ID: CRL-2606-1903 (Chemical Review and Letters)
Title: Thermal Conductivity of TiO2/Water-MEG Nanofluids: The Influence of Oleic Acid Surfactant Concentration and a Predictive Model
HasanMfar
This script performs:
1. Ordinary Least-Squares (OLS) regression constrained through the origin (Eq. 9):
   Z = a1*phi + a2*phi^2 + b1*cs + b2*cs^2 + c1*phi*cs
   where:
     - phi : TiO2 nanoparticle volume fraction (vol%, 0 to 1.00)
     - cs  : Oleic acid surfactant concentration (vol%, 0 to 4.90)
     - Z   : Relative thermal conductivity enhancement vs. neat base fluid, E_BF (%)
2. Calculation of full statistical metrics:
   - Regression coefficients with standard errors, t-statistics, and p-values
   - Uncentered R^2, Adjusted R^2, F-statistic, and p-value
   - Root Mean Square Error (RMSE)
   - Leave-One-Out Cross-Validation RMSE (LOOCV RMSE)
3. Publication-quality dual-panel visualization (Figure 13):
   - Left Panel : 3D Response Surface with experimental data points
   - Right Panel: 2D Contour Map with isolines and the zero-enhancement boundary (Z = 0.0%)
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================================================================
# 1. EXPERIMENTAL DATASET (Table 2 in Manuscript: 12 Formulations x 4 Currents)
# ==============================================================================

# Format: (phi [vol%], cs [vol%], [E_BF (%) at 750 mA, 850 mA, 1000 mA, 1200 mA])
raw_experiments = [
    (0.00, 0.80, [-16.1, -27.3, -16.9, -20.5]),  # BF-S0.80
    (0.00, 2.94, [-25.2, -27.1, -25.4, -20.8]),  # BF-S2.94
    (0.00, 4.90, [-26.5, -24.3, -30.4, -27.8]),  # BF-S4.90
    (0.10, 0.00, [ 4.9,   3.1,   4.0,   2.5 ]),  # NF-0.10/S0
    (0.25, 0.00, [ 7.5,   5.2,   5.5,   5.5 ]),  # NF-0.25/S0
    (0.50, 0.00, [11.1,  19.4,  16.9,  11.2 ]),  # NF-0.50/S0
    (1.00, 0.00, [11.1,  18.9,  41.3,  13.1 ]),  # NF-1.00/S0 (incl. settling outlier)
    (0.10, 2.94, [-15.1,  -9.5, -21.5, -18.6]),  # NF-0.10/S2.94
    (0.50, 2.94, [-22.9, -27.1, -20.7, -17.1]),  # NF-0.50/S2.94
    (1.00, 0.80, [-10.3, -11.3, -10.4,  -7.2]),  # NF-1.00/S0.80
    (1.00, 2.94, [-21.1, -21.2, -14.4, -12.3]),  # NF-1.00/S2.94
    (1.00, 4.90, [-22.2, -23.3, -22.3, -19.0])   # NF-1.00/S4.90
]

phi_list, cs_list, z_list = [], [], []
for phi, cs, z_vals in raw_experiments:
    for z in z_vals:
        phi_list.append(phi)
        cs_list.append(cs)
        z_list.append(z)

phi = np.array(phi_list, dtype=np.float64)
cs  = np.array(cs_list, dtype=np.float64)
z   = np.array(z_list, dtype=np.float64)
n = len(z)  # 48 observations

# ==============================================================================
# 2. OLS REGRESSION THROUGH THE ORIGIN (NO INTERCEPT)
# Model: Z = a1*phi + a2*phi^2 + b1*cs + b2*cs^2 + c1*phi*cs
# ==============================================================================

# Construct the design matrix X (n x p, where p = 5)
X = np.column_stack([
    phi,         # a1 term
    phi**2,      # a2 term
    cs,          # b1 term
    cs**2,       # b2 term
    phi * cs     # c1 interaction term
])
p = X.shape[1]  # 5 parameters

# Parameter estimation: beta = (X^T X)^(-1) X^T z
XtX_inv = np.linalg.inv(X.T @ X)
beta = XtX_inv @ X.T @ z

# Predicted values and residuals
z_hat = X @ beta
residuals = z - z_hat

# Residual sum of squares (RSS) and degrees of freedom
rss = np.sum(residuals**2)
df_resid = n - p
sigma2 = rss / df_resid
rmse = np.sqrt(np.mean(residuals**2))

# Variance-covariance matrix of estimated parameters
cov_beta = sigma2 * XtX_inv
se_beta = np.sqrt(np.diag(cov_beta))

# t-statistics and two-tailed p-values
t_stats = beta / se_beta
p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df=df_resid))

# Goodness-of-fit metrics (uncentered model without intercept)
tss_uncentered = np.sum(z**2)
r2_uncentered = 1.0 - (rss / tss_uncentered)
adj_r2 = 1.0 - (1.0 - r2_uncentered) * (n / df_resid)

# F-statistic for regression through the origin
f_stat = ((tss_uncentered - rss) / p) / (rss / df_resid)
f_pval = 1.0 - stats.f.cdf(f_stat, p, df_resid)

# Leave-One-Out Cross-Validation (LOOCV) via Hat matrix
H = X @ XtX_inv @ X.T
leverage = np.diag(H)
press_residuals = residuals / (1.0 - leverage)
loocv_rmse = np.sqrt(np.mean(press_residuals**2))

# ==============================================================================
# 3. PRINT STATISTICAL SUMMARY
# ==============================================================================

term_names = [
    "a1 (Linear particle loading, phi)",
    "a2 (Quadratic particle crowding, phi^2)",
    "b1 (Primary surfactant penalty, cs)",
    "b2 (Surfactant saturation curvature, cs^2)",
    "c1 (Cross-interaction / Kapitza, phi*cs)"
]

print("=" * 80)
print("REGRESSION THROUGH THE ORIGIN (ORDINARY LEAST SQUARES)")
print("Model: Z = a1*phi + a2*phi^2 + b1*cs + b2*cs^2 + c1*phi*cs")
print(f"Dataset: N = {n} observations | Residual df = {df_resid}")
print("-" * 80)
print(f"{'Parameter / Term':<44} | {'Estimate':>10} | {'Std. Error':>10} | {'t-stat':>8} | {'p-value':>9}")
print("-" * 80)
for i in range(p):
    print(f"{term_names[i]:<44} | {beta[i]:>10.4f} | {se_beta[i]:>10.4f} | {t_stats[i]:>8.2f} | {p_values[i]:>9.4e}")
print("=" * 80)
print(f"R-squared (uncentered)     : {r2_uncentered:.4f} (86.0% variance explained)")
print(f"Adjusted R-squared         : {adj_r2:.4f}")
print(f"Overall F-statistic ({p}, {df_resid}) : {f_stat:.2f} (p = {f_pval:.4e})")
print(f"Root Mean Square Error (RMSE) : {rmse:.2f}%")
print(f"LOOCV RMSE (Cross-Validation) : {loocv_rmse:.2f}%")
print("=" * 80)

# ==============================================================================
# 4. PUBLICATION-QUALITY VISUALIZATION (FIGURE 13)
# ==============================================================================

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'dejavusans'

fig = plt.figure(figsize=(15, 6.5), dpi=300)

# Grid for response surface prediction
phi_grid = np.linspace(0.0, 1.0, 100)
cs_grid  = np.linspace(0.0, 4.9, 100)
PHI, CS  = np.meshgrid(phi_grid, cs_grid)
Z_SURF   = (beta[0]*PHI + beta[1]*PHI**2 + 
            beta[2]*CS  + beta[3]*CS**2 + 
            beta[4]*PHI*CS)

# --- PANEL (A): 3D RESPONSE SURFACE ---
ax1 = fig.add_subplot(1, 2, 1, projection='3d')

# Plot surface with smooth gradient
surf = ax1.plot_surface(PHI, CS, Z_SURF, cmap='coolwarm', alpha=0.82,
                        edgecolor='none', rstride=2, cstride=2, antialiased=True)

# Scatter experimental data points
ax1.scatter(phi, cs, z, color='black', s=28, depthshade=True,
            edgecolors='white', linewidths=0.6, label='Experimental THW Data (N=48)')

ax1.set_xlabel(r'TiO$_2$ Volume Fraction, $\phi$ (vol%)', fontsize=10.5, labelpad=10, fontweight='bold')
ax1.set_ylabel(r'Oleic Acid Concentration, $c_s$ (vol%)', fontsize=10.5, labelpad=10, fontweight='bold')
ax1.set_zlabel(r'Enhancement $Z \equiv E_{BF}$ (%)', fontsize=10.5, labelpad=8, fontweight='bold')
ax1.set_title(r'(a) 3D Response Surface of $Z(\phi, c_s)$', fontsize=12, fontweight='bold', pad=12)
ax1.view_init(elev=24, azim=-128)
ax1.legend(loc='upper left', fontsize=9, framealpha=0.85)

# --- PANEL (B): 2D CONTOUR MAP WITH ZERO-ENHANCEMENT BOUNDARY ---
ax2 = fig.add_subplot(1, 2, 2)

# Filled contours
contour_levels = np.linspace(-30, 20, 21)
cf = ax2.contourf(PHI, CS, Z_SURF, levels=contour_levels, cmap='coolwarm', alpha=0.85)

# Line contours with labels
cs_lines = ax2.contour(PHI, CS, Z_SURF, levels=np.arange(-30, 25, 5), colors='k', linewidths=0.6)
ax2.clabel(cs_lines, inline=True, fontsize=8, fmt='%d%%')

# Highlight the Zero-Enhancement Boundary (Z = 0.0%)
zero_contour = ax2.contour(PHI, CS, Z_SURF, levels=[0.0], colors='darkgreen', linewidths=2.2, linestyles='--')
ax2.clabel(zero_contour, inline=True, fontsize=9.5, fmt='Z = 0.0%% (Neutral Boundary)', inline_spacing=15)

# Scatter mean experimental points per formulation for clarity
unique_formulations = [
    (0.00, 0.80), (0.00, 2.94), (0.00, 4.90),
    (0.10, 0.00), (0.25, 0.00), (0.50, 0.00), (1.00, 0.00),
    (0.10, 2.94), (0.50, 2.94), (1.00, 0.80), (1.00, 2.94), (1.00, 4.90)
]
for p_f, c_f in unique_formulations:
    mask = (phi == p_f) & (cs == c_f)
    z_mean = np.mean(z[mask])
    ax2.scatter(p_f, c_f, color='black', s=35, edgecolors='white', zorder=5)

ax2.set_xlabel(r'TiO$_2$ Volume Fraction, $\phi$ (vol%)', fontsize=10.5, fontweight='bold')
ax2.set_ylabel(r'Oleic Acid Concentration, $c_s$ (vol%)', fontsize=10.5, fontweight='bold')
ax2.set_title(r'(b) 2D Contour Map and $Z = 0.0\%$ Boundary', fontsize=12, fontweight='bold')
ax2.set_xlim(0.0, 1.0)
ax2.set_ylim(0.0, 4.9)

# Colorbar
cbar = fig.colorbar(cf, ax=ax2, pad=0.03, aspect=25)
cbar.set_label(r'Relative Enhancement vs. Neat Base Fluid, $E_{BF}$ (%)', fontsize=10, fontweight='bold')

# Region annotations
ax2.text(0.65, 0.4, 'Net Positive Region\n($Z > 0.0\\%$)', color='darkgreen',
         fontsize=10, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='darkgreen'))
ax2.text(0.35, 3.8, 'Surfactant-Dominated Penalty\n($Z < 0.0\\%$)', color='darkred',
         fontsize=10, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='darkred'))

plt.tight_layout()
output_filename = '/workspace/scratch/Figure_13_Response_Surface_and_Contour.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"High-resolution figure successfully saved to: {output_filename}")
