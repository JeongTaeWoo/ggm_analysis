import func
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA
from statsmodels.tsa.arima.model import ARIMA
from tqdm.autonotebook import tqdm

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

print("데이터 로딩 완료.")
print(f"남성 사망률 데이터 형태: {mortality_male_df.shape}")
print(f"여성 사망률 데이터 형태: {mortality_female_df.shape}")
print(f"사망률 데이터프레임의 마지막 연령: {mortality_male_df.index.values[-1]}")

# --- 2단계: 로그 사망률 변환 및 FPCA ---
log_mortality_male = np.log(mortality_male_df.values.T + 1e-10)
log_mortality_female = np.log(mortality_female_df.values.T + 1e-10)

# FPCA
fpca_male = FPCA(n_components=3)
fpca_male.fit(FDataGrid(data_matrix=log_mortality_male, grid_points=ages))
scores_male = fpca_male.transform(FDataGrid(data_matrix=log_mortality_male, grid_points=ages))

fpca_female = FPCA(n_components=3)
fpca_female.fit(FDataGrid(data_matrix=log_mortality_female, grid_points=ages))
scores_female = fpca_female.transform(FDataGrid(data_matrix=log_mortality_female, grid_points=ages))

print("\nFPCA 완료.")

# --- 3단계: 주성분 점수 시계열 예측 (ARIMA) ---
future_years = 7
forecast_years = [int(years[-1]) + i for i in range(1, future_years + 1)]
forecast_scores_male = np.zeros((future_years, fpca_male.n_components))
forecast_scores_female = np.zeros((future_years, fpca_female.n_components))

print("\nARIMA 모델을 사용하여 주성분 점수 예측 중...")
for i in tqdm(range(fpca_male.n_components), desc="남성 예측"):
    model = ARIMA(scores_male[:, i], order=(1, 1, 0))
    model_fit = model.fit()
    forecast_scores_male[:, i] = model_fit.forecast(steps=future_years)

for i in tqdm(range(fpca_female.n_components), desc="여성 예측"):
    model = ARIMA(scores_female[:, i], order=(1, 1, 0))
    model_fit = model.fit()
    forecast_scores_female[:, i] = model_fit.forecast(steps=future_years)

# --- 4단계: 미래 사망률 재구성 및 변환 ---
# 오류가 발생했던 부분을 수정했습니다.
# 평균 함수 데이터의 형태를 (99, 1)에서 (1, 99)로 변경하여 브로드캐스팅 오류를 해결합니다.
reconstructed_log_mortality_male = fpca_male.mean_.data_matrix[0].T + np.dot(forecast_scores_male, fpca_male.components_.data_matrix.squeeze())
reconstructed_mortality_male = np.exp(reconstructed_log_mortality_male)

reconstructed_log_mortality_female = fpca_female.mean_.data_matrix[0].T + np.dot(forecast_scores_female, fpca_female.components_.data_matrix.squeeze())
reconstructed_mortality_female = np.exp(reconstructed_log_mortality_female)

# --- 5단계: 결과 시각화 ---
print("\n결과를 시각화합니다.")
plt.figure(figsize=(12, 6))
plt.title('Male mortality forecast (FDM)')
plt.xlabel('Age')
plt.ylabel('Mortality Rate')
for col in mortality_male_df.columns:
    plt.plot(mortality_male_df.index, mortality_male_df[col], color='gray', alpha=0.5)
for i, year in enumerate(forecast_years):
    plt.plot(ages, reconstructed_mortality_male[i], linestyle='--', label=f'Forecast {year}')
plt.legend(['Observed', 'Forecast'])
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.title('Female mortality forecast (FDM)')
plt.xlabel('Age')
plt.ylabel('Mortality Rate')
for col in mortality_female_df.columns:
    plt.plot(mortality_female_df.index, mortality_female_df[col], color='gray', alpha=0.5)
for i, year in enumerate(forecast_years):
    plt.plot(ages, reconstructed_mortality_female[i], linestyle='--', label=f'Forecast {year}')
plt.legend(['Observed', 'Forecast'])
plt.grid(True)
plt.show()