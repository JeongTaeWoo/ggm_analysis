import func 
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
from skfda import FDataGrid
from skfda.exploratory.functional_data_analysis import FPCA
from statsmodels.tsa.arima.model import ARIMA
from itertools import product
from tqdm.autonotebook import tqdm

# --- 1단계: 데이터 불러오기 및 전처리 ---
# func.py의 load_life_table 함수를 사용하여 데이터 불러오기
# 편의를 위해 CSV 파일 내용을 직접 사용하지만, 실제로는 파일 경로를 사용합니다.
file_path_or_content = """age,title,1970,1971,1972,1973,1974,1975,1976,1977,1978,1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023
... (data omitted for brevity) ...
100세 이상,사망률(남자),1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
"""
csv_file_like = io.StringIO(file_path_or_content)

# 남성 데이터 로딩
years_male, ages_male, mortality_male = func.load_life_table(csv_file_like, '남자')

# 여성 데이터 로딩
csv_file_like.seek(0) # 파일을 다시 읽기 위해 포인터를 맨 앞으로 이동
years_female, ages_female, mortality_female = func.load_life_table(csv_file_like, '여자')


# --- 2단계: 로그 사망률 변환 및 FPCA ---
# 안정적인 로그 변환을 위해 매우 작은 값을 더합니다.
log_mortality_male = np.log(mortality_male + 1e-10)
log_mortality_female = np.log(mortality_female + 1e-10)

# 남성 데이터 FPCA
fd_male = FDataGrid(data_matrix=log_mortality_male.T, grid_points=ages_male)
fpca_male = FPCA(n_components=3)
fpca_male.fit(fd_male)
scores_male = fpca_male.transform(fd_male)

# 여성 데이터 FPCA
fd_female = FDataGrid(data_matrix=log_mortality_female.T, grid_points=ages_female)
fpca_female = FPCA(n_components=3)
fpca_female.fit(fd_female)
scores_female = fpca_female.transform(fd_female)

print("FPCA를 통해 남성 및 여성 로그 사망률의 주성분을 성공적으로 추출했습니다.")

# --- 3단계: 주성분 점수 시계열 예측 (ARIMA) ---
future_years = 7
forecast_years = [int(years_male[-1]) + i for i in range(1, future_years + 1)]
forecast_scores_male = np.zeros((future_years, fpca_male.n_components))
forecast_scores_female = np.zeros((future_years, fpca_female.n_components))

print("\nARIMA 모델을 사용하여 주성분 점수를 예측합니다.")
for i in tqdm(range(fpca_male.n_components), desc="남성 주성분 예측"):
    model = ARIMA(scores_male[:, i], order=(1, 1, 0))
    model_fit = model.fit()
    forecast_scores_male[:, i] = model_fit.forecast(steps=future_years)

for i in tqdm(range(fpca_female.n_components), desc="여성 주성분 예측"):
    model = ARIMA(scores_female[:, i], order=(1, 1, 0))
    model_fit = model.fit()
    forecast_scores_female[:, i] = model_fit.forecast(steps=future_years)

# --- 4단계: 미래 사망률 재구성 및 변환 ---
# 남성 미래 사망률 재구성
mean_func_male = fpca_male.mean_
reconstructed_log_mortality_male = mean_func_male.data_matrix[0] + np.dot(forecast_scores_male, fpca_male.components_.data_matrix.squeeze())
reconstructed_mortality_male = np.exp(reconstructed_log_mortality_male)

# 여성 미래 사망률 재구성
mean_func_female = fpca_female.mean_
reconstructed_log_mortality_female = mean_func_female.data_matrix[0] + np.dot(forecast_scores_female, fpca_female.components_.data_matrix.squeeze())
reconstructed_mortality_female = np.exp(reconstructed_log_mortality_female)

# --- 5단계: 결과 시각화 ---
print("\n결과를 시각화합니다.")
# 남성 사망률 시각화
plt.figure(figsize=(12, 6))
plt.title('남성 사망률 예측 (FDM)')
plt.xlabel('Age')
plt.ylabel('Mortality Rate')
for i, year in enumerate(years_male):
    plt.plot(ages_male, mortality_male[i], color='gray', alpha=0.5, label='Observed' if i == 0 else "")
for i, year in enumerate(forecast_years):
    plt.plot(ages_male, reconstructed_mortality_male[i], linestyle='--', label=f'Forecast {year}')
plt.legend()
plt.grid(True)
plt.show()

# 여성 사망률 시각화
plt.figure(figsize=(12, 6))
plt.title('여성 사망률 예측 (FDM)')
plt.xlabel('Age')
plt.ylabel('Mortality Rate')
for i, year in enumerate(years_female):
    plt.plot(ages_female, mortality_female[i], color='gray', alpha=0.5, label='Observed' if i == 0 else "")
for i, year in enumerate(forecast_years):
    plt.plot(ages_female, reconstructed_mortality_female[i], linestyle='--', label=f'Forecast {year}')
plt.legend()
plt.grid(True)
plt.show()