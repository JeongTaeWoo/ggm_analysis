#  Information measures and design issues in the study of mortality deceleration: findings for the gamma-Gompertz model 논문을 참고함

# frailty가 존재하는지 판별하는 테스트
# H0 : gamma = 0 ; H1 : gamma > 0
# 이때 Likelihood Ratio (우도비) 검정통계량 T = 2(L1 - L0)
# L(0,1)은 각 가설의 경우에서 발생하는 최대 로그우도값
# 본래는 Self & Liang, 1987의 연구처럼 혼합 카이제곱 분포를 사용해야 하지만, 
# 편의상 자유도 1인 카이제곱으로 사용, 카이제곱(1, 0.95) = 3.84


# 여기에서 frailty의 발생 가능성을 측정하는 beta도 있다.
# beta_n(sigma^2) ≈ 1 - Φ[ Φ^{-1}(1 - alpha) - sqrt(n) * (sigma^2 / kappa) ]
# sigma^2 = gamma

"""
| 기호                    | 설명                                                                      |
| ----------------------- | ----------------------------------------------------------------------- |
| beta_n(sigma^2)         | 표본 수 n일 때, frailty 분산 sigma^2에 대한 검정력 (즉, H₁이 맞을 때 H₀를 기각할 확률) |
| alpha                   | 유의수준 (보통 0.05 또는 0.01로 설정)                                              |
| Phi                     | 표준 정규분포의 누적분포함수(CDF)                                                    |
| Phi^{-1}(1 - alpha)     | 표준 정규분포의 임계값 (예: alpha = 0.05 → 약 1.6449)                               |
| n                       | 표본 크기 (사망자 수 또는 생존자 수 등)                                                |
| sigma^2                 | 실제 frailty 분산 값 (귀무가설은 sigma^2 = 0)                                  |
| kappa                   | fisher 정보행렬에서 sqrt(I^-1(theta)_3,3)                     |
"""

from scipy.optimize import differential_evolution, minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import chi2, norm
import numdifftools as nd # type: ignore

Dx = None; Ex = None

# 후보 경로들
file_path_candidates = [
    "C:/Users/pc/Desktop/project/65이상 생명표.xlsx",
    "C:/Users/tw010/Desktop/project/65이상 생명표.xlsx"
]

# 성공한 경로를 찾기
file_path = None
file_path = next((p for p in file_path_candidates if os.path.exists(p)), None)

if file_path is None:
    raise FileNotFoundError("생명표 파일을 찾을 수 없습니다. 경로를 확인하세요.")
else:
    print("경로 연결됨")
    df = pd.read_excel(file_path, sheet_name="Sheet1")

df_surv = df[df['title'] == '생존자(남자)'] # age 추출용이라 성별 상관없음
age_raw = pd.to_numeric(df_surv['age'], errors='coerce')
age = age_raw[:-1].reset_index(drop=True)


"""
해야할 것들 정리
1. ggm 최대우도 가져오기
2. gm 최대우도 가져오기
3. LRT 통계량 계산
4. 3.을 이용해서 p-value 계산
5. kappa 계산
6. beta 계산식
"""
def read_excel(sex, year) :
    # sex 인자에 따라 title 문자열 자동 생성
    surv_title = f"생존자({sex})"
    exp_title  = f"정지인구({sex})"
    # 생존자, 노출 DF 분리
    df_surv = df[df['title'] == surv_title]
    df_exp  = df[df['title'] == exp_title]
    
    # lx, Ex 불러오기
    age_raw = pd.to_numeric(df_surv['age'], errors='coerce')
    lx_raw  = pd.to_numeric(df_surv.get(year, []), errors='coerce')
    Ex_raw  = pd.to_numeric(df_exp.get(year, []), errors='coerce')

    # 1세 차분으로 Dx 계산
    age = age_raw[:-1].reset_index(drop=True)
    lx = lx_raw[:-1].reset_index(drop=True)
    lx_plus1 = lx_raw[1:].reset_index(drop=True)
    Ex = Ex_raw[:-1].reset_index(drop=True)
    Dx = lx - lx_plus1

    # 유효값 필터링
    valid = (~Dx.isna()) & (~Ex.isna()) & (Dx >= 0) & (Ex > 0)
    age = age[valid].reset_index(drop=True).values
    Dx = Dx[valid].reset_index(drop=True).values
    Ex = Ex[valid].reset_index(drop=True).values
    return Dx, Ex


def ggm_calc (params, age = age) :
    a, b, gamma, c = params
    log_num = np.log(a) + b * age
    log_denom = np.log1p((gamma * a / b) * (np.expm1(b * age))) 
    mu = np.exp(log_num - log_denom) + c
    log_mu = np.log(np.maximum(mu, 1e-10))
    logL = np.sum(Dx * log_mu - Ex * mu) # 가중치 일단 빼놨음

    return logL

# GM 모형
def neg_log_likelihood_gm(params, age, Dx, Ex):
    a, b, c = params
    mu = a * np.exp(b * age) + c
    mu = np.maximum(mu, 1e-10)  # 로그 안정화
    logL = np.sum(Dx * np.log(mu) - Ex * mu)
    return -logL

# GM 적합 함수
def gm_calc(age, Dx, Ex, bounds=[(1e-7, 0.01), (0.001, 0.2), (0, 0.01)]):
    # 초기값 설정 (약한 제약 포함)
    init_params = [1e-5, 0.05, 0.001]
    
    result = minimize(
        fun = neg_log_likelihood_gm,
        x0 = init_params,
        args = (age, Dx, Ex),
        bounds = bounds,
        method = 'L-BFGS-B'
    )
    
    if result.success:
        logL_gm = -result.fun
    return logL_gm    

def compute_LRT_stat(logL_alt, logL_null):
    T = 2 * (logL_alt - logL_null) #3.84보다 크면 p < 0.05로 기각 가능
    return T

def compute_p_value_LRT(T, method="chi2"):
    """
    LRT 통계량 T에 대한 p-value 계산
    method:
    - 'chi2'    : 일반적인 자유도 1의 카이제곱 분포 사용
    - 'mixture' : Self & Liang (1987) 방식, frailty 분산 검정용
    """
    if T <= 0:
        return 1.0  # 우도차가 음수거나 0이면, 복잡한 모델이 더 나쁨
    if method == "chi2":
        return 1 - chi2.cdf(T, df = 1)
    elif method == "mixture": # 귀무가설이 모수공간의 boundary에 있어서 사용함. df = 0인 카이제곱 분포(0으로 처리)와 반반 섞어서 계산한다
        return 0.5 * (1 - chi2.cdf(T, df = 1))
    else:
        raise ValueError("지원하지 않는 method입니다. 'chi2' 또는 'mixture' 중 선택하세요.")

def compute_power(n, gamma, kappa, alpha = 0.05):
    z_alpha = norm.ppf(1 - alpha)
    term = np.sqrt(n) * (gamma / kappa)
    power = 1 - norm.cdf(z_alpha - term)
    print("검정력:",100 * power,"%")
    return power

def summarize_LRT(logL_alt, logL_null, method = "chi2"):
    T = compute_LRT_stat(logL_alt, logL_null)
    p = compute_p_value_LRT(T, method=method)
    print("[LRT 결과 요약]")
    print(f"- 로그우도 (GGM): {logL_alt:.4f}")
    print(f"- 로그우도 (GM): {logL_null:.4f}")
    print(f"- 검정통계량 T: {T:.4f}")
    print(f"- p-value ({method}): {p:.5f}")
    if p > 0.1:
        print("→ 통계적으로 유의하지 않음")
    elif p > 0.05:
        print("→ 약한 증거 (추가 검토 필요)")
    elif p > 0.01:
        print("→ 유의함")
    elif p > 0.001:
        print("→ 매우 유의함")
    else:
        print("→ 극도로 유의함")
    return T, p

def compute_kappa(theta_hat, age, gamma_index = 2):
    """
    Fisher 정보행렬의 역행렬에서 gamma (또는 sigma^2)의 분산 추출
    theta_hat: 최적화된 파라미터 (a, b, gamma, c)
    gamma_index: gamma의 위치 (기본: index=2)
    """
    logL = lambda theta: ggm_calc(theta, age)
    hessian_func = nd.Hessian(logL)
    H = hessian_func(theta_hat)
    Fisher = -H
    Fisher_inv = np.linalg.inv(Fisher)
    kappa_squared = Fisher_inv[gamma_index, gamma_index]
    kappa = np.sqrt(kappa_squared)
    return kappa

# 2001, 남자 ggm 적합결과 : a = 0.00002294, b = 0.10586833, gamma = 0.11240457, c = 0.00108915 (0.00002355, 0.10554539, 0.11169777, 0.00102811)
Dx, Ex = read_excel(sex = "남자", year = 2001)

params_ggm = (0.00002764, 0.10355978, 0.10113280, 0.00039967) # (a, b, gamma, c)
max_ggm_logL = (-1) * ggm_calc(params = params_ggm)
max_gm_logL = (-1) * gm_calc(age = age, Dx = Dx, Ex = Ex)
summarize_LRT(max_ggm_logL, max_gm_logL)
kappa = compute_kappa(theta_hat = params_ggm, age = age)

compute_power(n = sum(Dx), gamma = params_ggm[2], kappa = kappa)
