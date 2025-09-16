import func
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

years, ages, df_mu, df_Dx, df_Ex = func.load_life_table(key = "kr", sex = "남자")

train_years = [str(y) for y in range(1970, 2010)]
forecast_years = [str(y) for y in range(2011, 2017)]

train_data = df_mu[train_years]
forecast_data = df_mu[forecast_years] 

log_df_mu = np.log(train_data) 

alpha_x  = log_df_mu.mean(axis = 1) # 각 연령별 로그사력 평균으로 alpha_x 추정
M_prime = log_df_mu.subtract(alpha_x, axis = 0) # 이 행렬에 SVD 수행

U, S, Vh = np.linalg.svd(M_prime, full_matrices = False)
s1 = S[0]
U1 = U[: , 0]
V1 = Vh[0 , :]

# LC모형의 정규화 제약 (sum(beta_x) = 1, sum(kappa_t) = 0)
beta_x = U1 / sum(U1)
kappa_t_raw = s1 * V1
kappa_t = kappa_t_raw - np.mean(kappa_t_raw)

alpha_x = alpha_x + beta_x * np.mean(kappa_t_raw) # alpha 보정

# kappa_t 시계열 데이터에 ARIMA(0,1,0) = random walk with drift 모델로 예측
model = ARIMA(kappa_t, order = (0, 1, 0), trend="t")
result = model.fit()

# 미래 kappa_t 예측
forecast_length = len(forecast_years) 
kappa_forecast = result.forecast(steps = forecast_length) 

# 미래 사망력 예측
alpha_x_matrix = np.tile(alpha_x, (forecast_length, 1)).T
beta_x_matrix = np.tile(beta_x, (forecast_length, 1)).T
kappa_forecast_matrix = np.tile(kappa_forecast, (len(beta_x), 1))

# 예측된 로그 사망력 계산
log_mx_forecast = alpha_x_matrix + beta_x_matrix * kappa_forecast_matrix
mx_forecast = np.exp(log_mx_forecast)

# 적합력, 예측력 평가 지표
mx_fit_true = train_data.values
#mx_fit_pred = 

mx_validation_true = forecast_data.values
mx_validation_pred = mx_forecast

mae = np.mean(np.abs(mx_fit_true))

mfe = np.mean(mx_validation_true - mx_validation_pred)
mafe = np.mean(np.abs(mx_validation_true - mx_validation_pred))
