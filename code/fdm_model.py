import pandas as pd
import func
import numpy as np
import matplotlib.pyplot as plt
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import BSplineBasis
from skfda.misc.regularization import L2Regularization
import pmdarima as pm
from tqdm.autonotebook import tqdm

# 경고 메시지 무시
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# --- 1단계: 데이터 불러오기 및 전처리 ---
filepath = "전연령 생명표.xlsx"
sex_male = '남자'
sex_female = '여자'

try:
    years, ages, mortality_male_df, _, _ = func.load_life_table(key='kr', sex=sex_male)
    _, _, mortality_female_df, _, _ = func.load_life_table(key='kr', sex=sex_female)

except ValueError as e:
    print(f"Error: {e}")
    exit()

# 논문과 동일하게 fitting period를 2010년까지로 설정
fitting_start_year = 1970
fitting_end_year = 2010
mortality_male_fitting = mortality_male_df.loc[:, str(fitting_start_year):str(fitting_end_year)]
mortality_female_fitting = mortality_female_df.loc[:, str(fitting_start_year):str(fitting_end_year)]

# 2011년부터 2023년까지의 실제값을 예측값과 비교하기 위해 별도 저장
forecast_start_year = fitting_end_year + 1
forecast_end_year = 2023
future_years = forecast_end_year - fitting_end_year
mortality_male_actual = mortality_male_df.loc[:, str(forecast_start_year):str(forecast_end_year)]
mortality_female_actual = mortality_female_df.loc[:, str(forecast_start_year):str(forecast_end_year)]

print("데이터 로딩 완료.")
print(f"남성 사망률 적합 데이터 형태: {mortality_male_fitting.shape}")
print(f"여성 사망률 적합 데이터 형태: {mortality_female_fitting.shape}")

# --- 2단계: 로그 사망률 변환 및 평활화 ---
log_mortality_male = np.log(mortality_male_fitting.values.T + 1e-10)
log_mortality_female = np.log(mortality_female_fitting.values.T + 1e-10)

# B-스플라인 기저 함수를 사용하여 평활화
n_basis = 20
spline_basis = BSplineBasis(n_basis=n_basis, domain_range=(ages.min(), ages.max()))
lambda_val = 0.01

smoothed_data_male = FDataGrid(data_matrix=log_mortality_male, grid_points=ages).to_basis(
    basis=spline_basis,
    regularization=L2Regularization(regularization_parameter=lambda_val)
)
smoothed_data_female = FDataGrid(data_matrix=log_mortality_female, grid_points=ages).to_basis(
    basis=spline_basis,
    regularization=L2Regularization(regularization_parameter=lambda_val)
)

# --- 3단계: 함수적 주성분분석(FPCA) ---
fpca_male = FPCA(n_components=3)
fpca_male.fit(smoothed_data_male)
scores_male = fpca_male.transform(smoothed_data_male)

fpca_female = FPCA(n_components=3)
fpca_female.fit(smoothed_data_female)
scores_female = fpca_female.transform(smoothed_data_female)

print("\nFPCA 완료.")

# --- 4단계: 주성분 점수 시계열 예측 (ARIMA) ---
forecast_years = [i for i in range(forecast_start_year, forecast_end_year + 1)]
forecast_scores_male = np.zeros((future_years, fpca_male.n_components))
forecast_scores_female = np.zeros((future_years, fpca_female.n_components))

print("\nARIMA 모델을 사용하여 주성분 점수 예측 중...")
for i in tqdm(range(fpca_male.n_components), desc="남성 예측"):
    auto_model = pm.auto_arima(scores_male[:, i],
                                start_p=1, start_q=1,
                                max_p=3, max_q=3,
                                seasonal=False,
                                stepwise=True,
                                suppress_warnings=True,
                                D=1)
    forecast_scores_male[:, i] = auto_model.predict(n_periods=future_years)

for i in tqdm(range(fpca_female.n_components), desc="여성 예측"):
    auto_model = pm.auto_arima(scores_female[:, i],
                                start_p=1, start_q=1,
                                max_p=3, max_q=3,
                                seasonal=False,
                                stepwise=True,
                                suppress_warnings=True,
                                D=1)
    forecast_scores_female[:, i] = auto_model.predict(n_periods=future_years)

# --- 5단계: 미래 사망률 재구성 및 변환 ---
reconstructed_log_mortality_male = fpca_male.mean_.to_grid(grid_points=ages).data_matrix[0].T + np.dot(forecast_scores_male, fpca_male.components_.to_grid(grid_points=ages).data_matrix.squeeze())
reconstructed_mortality_male = np.exp(reconstructed_log_mortality_male)

reconstructed_log_mortality_female = fpca_female.mean_.to_grid(grid_points=ages).data_matrix[0].T + np.dot(forecast_scores_female, fpca_female.components_.to_grid(grid_points=ages).data_matrix.squeeze())
reconstructed_mortality_female = np.exp(reconstructed_log_mortality_female)

# --- 6단계: 예측 정확도 측정 (MFE, MAFE, MAE) ---
log_mortality_male_actual = np.log(mortality_male_actual.values.T + 1e-10)
log_mortality_female_actual = np.log(mortality_female_actual.values.T + 1e-10)

# 남성 오차 계산
mfe_male = np.mean(reconstructed_log_mortality_male - log_mortality_male_actual)
mafe_male = np.mean(np.abs(reconstructed_log_mortality_male - log_mortality_male_actual))
mae_male = np.mean(np.abs(reconstructed_log_mortality_male - log_mortality_male_actual))

# 여성 오차 계산
mfe_female = np.mean(reconstructed_log_mortality_female - log_mortality_female_actual)
mafe_female = np.mean(np.abs(reconstructed_log_mortality_female - log_mortality_female_actual))
mae_female = np.mean(np.abs(reconstructed_log_mortality_female - log_mortality_female_actual))

print("\n--- Forecast Error Analysis (2011-2023) ---")
print(f"Male Mean Forecast Error (MFE): {mfe_male:.4f}")
print(f"Male Mean Absolute Forecast Error (MAFE): {mafe_male:.4f}")
print(f"Male Mean Absolute Error (MAE): {mae_male:.4f}")
print(f"Female Mean Forecast Error (MFE): {mfe_female:.4f}")
print(f"Female Mean Absolute Forecast Error (MAFE): {mafe_female:.4f}")
print(f"Female Mean Absolute Error (MAE): {mae_female:.4f}")

# --- 7단계: 결과 시각화 ---
print("\nVisualizing results.")
colors = plt.cm.plasma(np.linspace(0, 1, mortality_male_fitting.shape[1]))
forecast_colors = plt.cm.viridis(np.linspace(0, 1, future_years))

# 남성 로그 사망률 시각화
plt.figure(figsize=(12, 6))
plt.title('Male Log Mortality Rate Forecast (FDM)')
plt.xlabel('Age')
plt.ylabel('Log Mortality Rate')
for i, col in enumerate(mortality_male_fitting.columns):
    # 관측 데이터: 옅은 회색의 얇은 선
    plt.plot(mortality_male_fitting.index, np.log(mortality_male_fitting[col].values), color='gray', alpha=0.3, linewidth=1)
for i, year in enumerate(forecast_years):
    # 예측 데이터: 굵은 점선
    plt.plot(ages, reconstructed_log_mortality_male[i], linestyle='--', color=forecast_colors[i], linewidth=2, label=f'Forecast {year}')
# 실제 데이터: 굵은 빨간색 실선
plt.plot(mortality_male_actual.index, np.log(mortality_male_actual.values), color='red', linestyle='-', linewidth=2, label='Actual Curves')
plt.legend(['Observed Curves'] + [f'Forecast {year}' for year in forecast_years] + ['Actual Curves'])
plt.grid(True)
plt.show()

# 여성 로그 사망률 시각화
plt.figure(figsize=(12, 6))
plt.title('Female Log Mortality Rate Forecast (FDM)')
plt.xlabel('Age')
plt.ylabel('Log Mortality Rate')
for i, col in enumerate(mortality_female_fitting.columns):
    # 관측 데이터: 옅은 회색의 얇은 선
    plt.plot(mortality_female_fitting.index, np.log(mortality_female_fitting[col].values), color='gray', alpha=0.3, linewidth=1)
for i, year in enumerate(forecast_years):
    # 예측 데이터: 굵은 점선
    plt.plot(ages, reconstructed_log_mortality_female[i], linestyle='--', color=forecast_colors[i], linewidth=2, label=f'Forecast {year}')
# 실제 데이터: 굵은 빨간색 실선
plt.plot(mortality_female_actual.index, np.log(mortality_female_actual.values), color='red', linestyle='-', linewidth=2, label='Actual Curves')
plt.legend(['Observed Curves'] + [f'Forecast {year}' for year in forecast_years] + ['Actual Curves'])
plt.grid(True)
plt.show()