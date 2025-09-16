"""
Coherent Functional Demographic Model (FDM)
- Weighted penalized spline smoothing (per year across ages)
- Product–Ratio decomposition on log scale
- FPCA with scikit-fda
- auto_arima with pmdarima

입력:
    mu_male  : pd.DataFrame (index=years, columns=ages)  - 남성 사망률
    mu_female: pd.DataFrame (index=years, columns=ages)  - 여성 사망률

출력:
    dict with keys:
        'forecast_male_mu', 'forecast_female_mu' : 예측 사망률 (원 스케일)
        'forecast_male_log', 'forecast_female_log' : 예측 로그 사망률
        'forecast_p_log', 'forecast_r_log' : 공통/차이 (product-ratio, log)
        'fpca_p', 'fpca_r' : 학습된 FPCA 객체
        'scores_p', 'scores_r' : 학습 점수 (행: 연도, 열: 성분)
        'forecast_scores_p', 'forecast_scores_r' : 예측 점수
        'smoothed_y_male', 'smoothed_y_female' : 평활화된 로그 사망률 (관측 구간)
        'metrics': 테스트 구간 평가 지표 (MAE, mfe)

주의:
    - scikit-fda, pmdarima, scipy 가 필요합니다.
    - smoothing 은 2-pass (unweighted → variance smoothing → weighted) 방식 간단 구현입니다.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import func
from scipy.interpolate import UnivariateSpline
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA
import pmdarima as pm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import warnings
import seaborn as sns
warnings.filterwarnings('ignore', category=FutureWarning)

# ------------------------------
# Helper: 1D spline smoothing with optional weights
# ------------------------------

def _smooth_1d(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None,
                k: int = 3, s: Optional[float] = None) -> np.ndarray:
    m = np.isfinite(y)
    x_m = x[m]
    y_m = y[m]
    if w is not None:
        w_m = np.clip(w[m], 1e-8, np.inf)
    else:
        w_m = None
    if x_m.size < (k + 1):
        return np.interp(x, x_m, y_m, left=y_m[0], right=y_m[-1])
    spl = UnivariateSpline(x_m, y_m, w=w_m, k=k, s=s)
    return spl(x)


# ------------------------------
# Panel smoother (across ages, per year) with two-pass weighting
# ------------------------------

def smooth_panel_weighted_log(mu_df: pd.DataFrame,
                              k: int = 3,
                              s_unweighted: Optional[float] = None,
                              s_var: Optional[float] = None,
                              s_weighted: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ages = np.asarray(mu_df.columns, dtype=float)
    years = mu_df.index
    n_age = ages.size
    if s_unweighted is None:
        s_unweighted = n_age
    if s_var is None:
        s_var = n_age
    if s_weighted is None:
        s_weighted = n_age

    y_log = np.log(mu_df.clip(lower=1e-12))
    f0_list, sigma2_list, f_list = [], [], []

    for t in years:
        y = y_log.loc[t].values.astype(float)
        f0 = _smooth_1d(ages, y, w=None, k=k, s=s_unweighted)
        r2 = np.maximum((y - f0) ** 2, 1e-10)
        sigma2 = _smooth_1d(ages, r2, w=None, k=k, s=s_var)
        sigma2 = np.maximum(sigma2, 1e-10)
        f = _smooth_1d(ages, y, w=1.0/sigma2, k=k, s=s_weighted)
        f0_list.append(f0)
        sigma2_list.append(sigma2)
        f_list.append(f)

    f_df = pd.DataFrame(np.vstack(f_list), index=years, columns=ages)
    sigma2_df = pd.DataFrame(np.vstack(sigma2_list), index=years, columns=ages)
    return f_df, sigma2_df


# ------------------------------
# FPCA + auto_arima pipeline (product–ratio on log scale)
# ------------------------------

@dataclass
class CoherentFDMPipeline:
    n_components_p: int = 1
    n_components_r: int = 1
    k_spline: int = 3
    s_unweighted: Optional[float] = None
    s_var: Optional[float] = None
    s_weighted: Optional[float] = None
    start_p: int = 0
    start_q: int = 0
    max_p: int = 3
    max_q: int = 3
    seasonal: bool = False
    stepwise: bool = True
    suppress_warnings: bool = True

    def fit_forecast(self, mu_male: pd.DataFrame, mu_female: pd.DataFrame,
                    forecast_years: Optional[int] = None,
                    train_years: Optional[Tuple[int,int]] = None,
                    test_years: Optional[Tuple[int,int]] = None) -> Dict[str, object]:
        
        mu_male.index = mu_male.index.astype(int)
        mu_female.index = mu_female.index.astype(int)
        mu_male.columns = mu_male.columns.astype(int)
        mu_female.columns = mu_female.columns.astype(int)

        mu_male_full, mu_female_full = mu_male.copy(), mu_female.copy()
        if train_years is not None:
            mu_male = mu_male.loc[train_years[0]:train_years[1]]
            mu_female = mu_female.loc[train_years[0]:train_years[1]]

        # 1) Weighted smoothing
        fM_log, sigma2M = smooth_panel_weighted_log(mu_male, k=self.k_spline,
            s_unweighted=self.s_unweighted, s_var=self.s_var, s_weighted=self.s_weighted)
        fF_log, sigma2F = smooth_panel_weighted_log(mu_female, k=self.k_spline,
            s_unweighted=self.s_unweighted, s_var=self.s_var, s_weighted=self.s_weighted)

        # 2) Product–Ratio decomposition
        p_log = 0.5 * (fM_log + fF_log)
        r_log = 0.5 * (fM_log - fF_log)
        ages = np.asarray(p_log.columns, dtype=float)
        years = p_log.index.values

        # 3) FPCA
        fdata_p = FDataGrid(data_matrix=p_log.values, grid_points=ages)
        fpca_p = FPCA(n_components=self.n_components_p)
        fpca_p.fit(fdata_p)
        scores_p = fpca_p.transform(fdata_p)

        fdata_r = FDataGrid(data_matrix=r_log.values, grid_points=ages)
        fpca_r = FPCA(n_components=self.n_components_r)
        fpca_r.fit(fdata_r)
        scores_r = fpca_r.transform(fdata_r)

        # 4) auto_arima
        def _forecast_scores(scores: np.ndarray, nsteps: int) -> np.ndarray:
            if scores.ndim == 1:
                scores = scores[:, None]
            out = []
            for j in range(scores.shape[1]):
                model = pm.auto_arima(scores[:, j],
                    start_p=self.start_p, start_q=self.start_q,
                    max_p=self.max_p, max_q=self.max_q,
                    seasonal=self.seasonal, stepwise=self.stepwise,
                    suppress_warnings=self.suppress_warnings,
                    d=None, D=0)
                out.append(model.predict(n_periods=nsteps))
            return np.column_stack(out) if out else np.zeros((nsteps, 0))

        if forecast_years is None and (train_years and test_years):
            forecast_years = test_years[1] - train_years[1]

        forecast_scores_p = _forecast_scores(scores_p, forecast_years)
        forecast_scores_r = _forecast_scores(scores_r, forecast_years)

        # 5) Reconstruct forecasts
        comp_p = np.atleast_2d(fpca_p.components_.to_grid(grid_points=ages).data_matrix.squeeze())
        mean_p = fpca_p.mean_.to_grid(grid_points=ages).data_matrix[0].ravel()
        comp_r = np.atleast_2d(fpca_r.components_.to_grid(grid_points=ages).data_matrix.squeeze())
        mean_r = fpca_r.mean_.to_grid(grid_points=ages).data_matrix[0].ravel()

        p_log_fore = forecast_scores_p @ comp_p + mean_p
        r_log_fore = forecast_scores_r @ comp_r + mean_r
        male_log_fore, female_log_fore = p_log_fore + r_log_fore, p_log_fore - r_log_fore
        male_mu_fore, female_mu_fore = np.exp(male_log_fore), np.exp(female_log_fore)

        last_year = int(mu_male.index[-1])
        future_index = pd.Index(range(last_year+1, last_year+1+forecast_years), name=mu_male.index.name)
        forecast_male_log = pd.DataFrame(male_log_fore, index=future_index, columns=ages)
        forecast_female_log = pd.DataFrame(female_log_fore, index=future_index, columns=ages)
        forecast_male_mu = pd.DataFrame(male_mu_fore, index=future_index, columns=ages)
        forecast_female_mu = pd.DataFrame(female_mu_fore, index=future_index, columns=ages)

        # 6) 평가 지표 (MAE, MFE)
        metrics = {}
        if test_years is not None:
            te_start, te_end = test_years
            y_true_m = mu_male_full.loc[te_start:te_end]
            y_true_f = mu_female_full.loc[te_start:te_end]
            y_pred_m = forecast_male_mu.loc[te_start:te_end] 
            y_pred_f = forecast_female_mu.loc[te_start:te_end]
            idx, cols = y_true_m.index.intersection(y_pred_m.index), y_true_m.columns.intersection(y_pred_m.columns)
            y_true_m, y_pred_m = y_true_m.loc[idx, cols], y_pred_m.loc[idx, cols]
            y_true_f, y_pred_f = y_true_f.loc[idx, cols], y_pred_f.loc[idx, cols]
            metrics['mae_male'] = mean_absolute_error(y_true_m.values.ravel(), y_pred_m.values.ravel())
            metrics['mfe_male'] = np.mean(y_true_m.values.ravel() - y_pred_m.values.ravel())
            metrics['mae_female'] = mean_absolute_error(y_true_f.values.ravel(), y_pred_f.values.ravel())
            metrics['mfe_female'] = np.mean(y_true_f.values.ravel() - y_pred_f.values.ravel())
            print(f"\n--- Forecast Error Analysis ({te_start}-{te_end}) ---")
            print(f"Male   MAE: {metrics['mae_male']:.6f}, MFE: {metrics['mfe_male']:.6f}")
            print(f"Female MAE: {metrics['mae_female']:.6f}, MFE: {metrics['mfe_female']:.6f}")

        return {
            'forecast_male_mu': forecast_male_mu,
            'forecast_female_mu': forecast_female_mu,
            'forecast_male_log': forecast_male_log,
            'forecast_female_log': forecast_female_log,
            'forecast_p_log': pd.DataFrame(p_log_fore, index=future_index, columns=ages),
            'forecast_r_log': pd.DataFrame(r_log_fore, index=future_index, columns=ages),
            'fpca_p': fpca_p, 'fpca_r': fpca_r,
            'scores_p': scores_p, 'scores_r': scores_r,
            'forecast_scores_p': forecast_scores_p,
            'forecast_scores_r': forecast_scores_r,
            'smoothed_y_male': fM_log,
            'smoothed_y_female': fF_log,
            'sigma2_male': sigma2M,
            'sigma2_female': sigma2F,
            'metrics': metrics
        }


# ------------------------------
# Quick plotting helpers
# ------------------------------

def plot_age_traces(observed: pd.DataFrame, forecast: pd.DataFrame, ages: Tuple[int, int] = (60, 80), title_prefix: str = "Male"):
    if observed.columns.dtype == object and str(observed.columns[0]).isdigit():
        observed = observed.T
    if forecast.columns.dtype == object and str(forecast.columns[0]).isdigit():
        forecast = forecast.T
    observed.index = observed.index.astype(int)
    forecast.index = forecast.index.astype(int)
    a1, a2 = ages
    plt.figure(figsize=(11, 5))
    plt.plot(observed.index, observed[a1], label=f'Observed age {a1}')
    plt.plot(observed.index, observed[a2], label=f'Observed age {a2}')
    plt.plot(forecast.index, forecast[a1], '--', label=f'Forecast age {a1}')
    plt.plot(forecast.index, forecast[a2], '--', label=f'Forecast age {a2}')
    plt.title(f"{title_prefix} mortality: observed vs. forecast")
    plt.xlabel('Year')
    plt.ylabel('Mortality rate')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_heatmap(df: pd.DataFrame, title: str):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, cmap='viridis', cbar_kws={'label': 'Mortality Rate'})
    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Year')
    plt.show()

# ------------------------------
# New plotting function for curve visualization
# ------------------------------
def plot_curves(observed_log_df: pd.DataFrame, forecast_log_df: pd.DataFrame, title_prefix: str):
    ages = observed_log_df.columns
    plt.figure(figsize=(12, 6))
    plt.title(f'{title_prefix} Log Mortality Rate Forecast (Coherent FDM)')
    plt.xlabel('Age')
    plt.ylabel('Log Mortality Rate') # y축 라벨을 Log Mortality Rate로 변경
    
    # Plot historical curves
    for year in observed_log_df.index:
        plt.plot(ages, observed_log_df.loc[year], color='gray', alpha=0.3, linewidth=1) # np.exp() 제거
    
    # Plot forecast curves
    forecast_years = forecast_log_df.index
    forecast_colors = plt.cm.viridis(np.linspace(0, 1, len(forecast_years)))
    for i, year in enumerate(forecast_years):
        plt.plot(ages, forecast_log_df.loc[year], linestyle='--', color=forecast_colors[i], linewidth=2, label=f'Forecast {year}')
    
    # Add legends for clarity
    handles = [plt.Line2D([0], [0], color='gray', alpha=0.3, linewidth=1, label='Historical Curves')]
    for i, year in enumerate(forecast_years):
        handles.append(plt.Line2D([0], [0], linestyle='--', color=forecast_colors[i], linewidth=2, label=f'Forecast {year}'))
    plt.legend(handles=handles)
    
    plt.grid(True)
    plt.show()

# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    years, ages, mu_male, Dx_male, Ex_male = func.load_life_table('kr', '남자')
    years, ages, mu_female, Dx_female, Ex_female = func.load_life_table('kr', '여자')
    mu_male = mu_male.T
    mu_female = mu_female.T

    pipe = CoherentFDMPipeline(n_components_p=1, n_components_r=2)
    out = pipe.fit_forecast(mu_male, mu_female,
                            train_years=(1970, 2010),
                            test_years=(2011, 2016))
    
    # Visualize mortality curves for all ages in the style of fdm_model.py
    plot_curves(out['smoothed_y_male'], out['forecast_male_log'], title_prefix='Male')
    plot_curves(out['smoothed_y_female'], out['forecast_female_log'], title_prefix='Female')

    # Optional: You can still use the previous plotting functions if you need them
    # plot_age_traces(mu_male, out['forecast_male_mu'], ages=(90, 99), title_prefix='Male')
    # plot_heatmap(out['forecast_male_mu'], 'Male Mortality - Forecast (2011-2016)')