import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import func  # func.py 파일을 그대로 import

def fit_and_forecast_fdm(sex='남자', forecast_years=10):
    """
    FDM 모형을 적용하여 사망률을 예측하고 결과를 시각화합니다.
    func.py의 load_life_table 함수를 사용하여 데이터를 로드합니다.
    """
    log_mortality_data = load_mortality_data_from_func(sex=sex)
    
    if log_mortality_data is None:
        print("데이터 로딩에 실패했습니다. 파일 경로와 형식을 확인해주세요.")
        return

    # FDM 모형의 2, 3, 4단계 구현
    # 2. 평균 함수 추정
    mean_function = log_mortality_data.mean(axis=0)

    # 3. 주성분 분석 (FPCA)
    deviations = log_mortality_data.sub(mean_function, axis=1)
    
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(deviations)
    functional_pcs = pca.components_
    principal_component_scores = pd.DataFrame(principal_components, index=deviations.index)
    
    # 4. 시계열 예측 (ARIMA)
    forecasted_scores = pd.DataFrame()
    for i in range(principal_component_scores.shape[1]):
        model = ARIMA(principal_component_scores.iloc[:, i], order=(1, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_years)
        forecasted_scores[i] = forecast
        
    forecasted_deviations = forecasted_scores @ functional_pcs
    forecasted_log_mortality = forecasted_deviations.add(mean_function, axis=1)
    
    # 결과 정리
    current_year = max(pd.to_numeric(log_mortality_data.index))
    forecast_index = pd.RangeIndex(start=current_year + 1, stop=current_year + 1 + forecast_years)
    forecasted_log_mortality.index = forecast_index
    
    # 예측된 로그 사망률을 사망률로 변환
    forecasted_mortality = np.exp(forecasted_log_mortality)
    
    print(f"{sex} 사망률 예측 결과:")
    print(forecasted_mortality)
    
    original_mortality = np.exp(log_mortality_data)
    
    plt.figure(figsize=(12, 8))
    ages_to_plot = [20, 40, 60, 80, 90]
    
    for age in ages_to_plot:
        if age in original_mortality.columns:
            combined_data = pd.concat([original_mortality[age], forecasted_mortality[age]])
            combined_data.plot(label=f'Age {age}', marker='o', linestyle='-')
            
    plt.title(f'FDM Mortality Forecast for {sex}')
    plt.xlabel('Year')
    plt.ylabel('Mortality Rate')
    plt.legend()
    plt.grid(True)
    plt.show()

def load_mortality_data_from_func(sex='남자'):
    """
    func.py의 load_life_table 함수를 사용하여 사망률 데이터를 로드합니다.
    """
    # func.py에 정의된 df를 직접 사용
    if func.df is None:
        return None
        
    # 데이터는 행과 열이 뒤바뀐 구조이므로, 미리 전처리
    data = func.df.set_index('title').drop(['age'], axis=1)
    
    years = data.columns
    ages = pd.to_numeric(func.df[func.df['title'] == 'age'].iloc[0, 2:].values)
    
    log_mortality_rates = []
    
    for year in years:
        # func.py의 load_life_table 함수를 호출
        # 함수가 반환하는 순서를 그대로 따릅니다: q_x, l_x, d_x, year, age
        # FDM 모형 구현에 필요한 d_x (사망자 수)와 l_x (생존자 수)를 사용
        
        q_x, l_x, d_x, _, _ = func.load_life_table(year, sex)
        
        # Dx와 Ex는 단일 값이므로, 전체 연령대의 데이터가 필요하다면
        # func.py의 df를 직접 활용하는 것이 더 효율적입니다.
        
        # 여기서는 func.py에서 로드된 df를 직접 사용해 전체 데이터를 가져옵니다.
        # 이 부분이 func.py를 그대로 사용하면서 FDM 모형에 필요한
        # 전체 데이터를 로드하는 가장 합리적인 방법입니다.
        
        if sex == '남자':
            Dx = data.loc['사망자(남자)']
            Ex = data.loc['정지인구(남자)']
        else:
            Dx = data.loc['사망자(여자)']
            Ex = data.loc['정지인구(여자)']
            
    # 사망자 수와 정지인구 수로 사망률 계산
    mortality = Dx.values / Ex.values
    log_mortality = np.log(mortality)

    return pd.DataFrame([log_mortality], index=years, columns=ages).T

def smooth_mortality_rates(log_mortality_rates):
    """B-spline을 사용하여 로그 사망률 데이터를 평활화합니다."""
    smoothed_rates = log_mortality_rates.copy()
    ages = smoothed_rates.columns.values
    
    for year in smoothed_rates.index:
        spline = UnivariateSpline(ages, smoothed_rates.loc[year], s=10)
        smoothed_rates.loc[year] = spline(ages)
        
    return smoothed_rates

# FDM 모형 구현 및 예측 실행
fit_and_forecast_fdm(sex='남자')