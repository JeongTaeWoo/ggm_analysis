from pathlib import Path
import random
from scipy.optimize import differential_evolution, minimize, dual_annealing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import trange
import xlsxwriter

age = np.arange(65, 100, dtype = float)  # 65세부터 99세까지 기본값 설정
Dx = None; Ex = None

base_dir = Path(__file__).resolve().parent

# 생명표 읽기
life_table_path = base_dir / "65이상 생명표.xlsx"
df = pd.read_excel(life_table_path, sheet_name="Sheet1")

output_path_batch = base_dir / "적합 결과.xlsx"
output_path_weight = base_dir / "가중치 측정 결과.xlsx"

def load_excel(year, sex, df = df) :
    df_surv = df[df['title'] == '생존자(남자)'] # age 추출용이라 성별 상관없음
    age_raw = pd.to_numeric(df_surv['age'], errors='coerce')
    age = age_raw[:-1].reset_index(drop=True)

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
    observed_mu = Dx/Ex
    
    return year, sex, Dx, Ex, age, observed_mu

# --- result 형태 생성 함수 ---
class SimpleResult:
    def __init__(self, x):
        self.x = x

def result_maker(a, b, gamma, c):
    return SimpleResult([a, b, gamma, c])

# TODO weight 여러종류 쓸 꺼면 class도 생각해볼만 함
def weight_sigmoid(age, center = 90, scale = 3, max_weight = 10):
    """
    중심(center) 기준으로 sigmoid 함수 형태의 가중치를 부여
    center 이후로 점점 가중치가 커짐
    scale이 작아질수록 가중치가 커짐
    """
    return 1 + (max_weight - 1) / (1 + np.exp(-(age - center) / scale))

def weight_rmse(y_obs, y_fit, age, center = 90, scale = 3, max_weight = 10):
    """
    sigmoid 가중치를 기반으로 예측값과 실제값 사이의 오차를 강조하여 계산
    """
    weights = weight_sigmoid(age, center = center, scale = scale, max_weight = max_weight)
    return np.sqrt(np.mean(weights * (y_obs - y_fit)**2))

def assess_fit_rmse(params, age, Dx, Ex, center = 90, scale = 3, max_weight = 10):
    """
    모델 파라미터와 age, Dx, Ex를 입력받아 RMSE 평가
    """
    a, b, gamma, c = params
    log_num = np.log(a) + b * age
    log_denom = np.log1p((gamma * a / b) * (np.expm1(b * age))) 
    fitted_mu = np.exp(log_num - log_denom) + c
    observed = Dx / Ex  # 중앙 사망률 (observed)
    
    return weight_rmse(observed, fitted_mu, age, center=center, scale=scale, max_weight=max_weight)


# 로그우도 함수 정의
def make_neg_log_likelihood(Dx, Ex, age, weight_func = None, weight_params = None) :
    def neg_log_likelihood(params):
        a, b, gamma, c = params
        
        # 로그 분모·분자를 쓰는 방법 (로그-합-지수 기법)
        log_num = np.log(a) + b * age
        log_denom = np.log1p((gamma * a / b) * (np.expm1(b * age))) 
        # log1p = log(1 + x), expm1 = exp(x)- 1 ; 둘다 반올림오차 줄여줌
        mu = np.exp(log_num - log_denom) + c
        
        # 일반 MLE
        # logL = np.sum(Dx * np.log(mu) - Ex * mu)
        
        # 가중치 계산 (기본은 모두 1)
        if weight_func :
            w_params = weight_params if weight_params is not None else {}
            weights = weight_func(age, **w_params)
        else:
            weights = np.ones_like(age)
        
        # 가중 로그우도 계산
        log_mu = np.log(np.maximum(mu, 1e-10))
        logL = np.sum(weights * (Dx * log_mu - Ex * mu))
        
        return -logL
    
    return neg_log_likelihood

def run_optimizer(opt_func, neg_log_likelihood, bounds = None, init_params = None) :
    if opt_func == "differential_evolution" :
        return differential_evolution(
            func = neg_log_likelihood,
            bounds = bounds,
            seed = None,
            maxiter = 500,
            polish = False,
            popsize = 60,
            mutation = (0.5, 1),
            recombination = 0.7,
            updating = "immediate",
            strategy = "best1bin"
        )

    elif opt_func == "minimize":
        if init_params is None:
            raise ValueError("minimize를 사용할 경우 init_params 필요")
        return minimize(
            fun = neg_log_likelihood,
            x0 = init_params,
            bounds = bounds,
            method = "L-BFGS-B"
        )
    elif opt_func == "dual_annealing":
        return dual_annealing(
            func = neg_log_likelihood,
            bounds = bounds,
            maxiter = 1000,
            initial_temp = 5230.0, # init = 5230.0
            visit = 2.62, # init = 2.62
            seed = None,
            local_search_options = {
                'method' : 'L-BFGS-B'
            }
            #,x0 = [0.0001, 0.1, 0.08, 0.0005]
        )
    
    else:
        raise ValueError("최적화 함수 오류")
    

# GM 모형
def neg_log_likelihood_gm(params, age, Dx, Ex):
    a, b, c = params
    mu = a * np.exp(b * age) + c
    mu = np.maximum(mu, 1e-10)  # 로그 안정화
    logL = np.sum(Dx * np.log(mu) - Ex * mu)
    return -logL

# GM 적합 함수
def fit_gm(age, Dx, Ex, bounds=[(0, 0.0005), (0.01, 0.5), (0, 0.05)]):
    # 초기값 설정 (약한 제약 포함)
    init_params = [0.00005, 0.1, 0.00005]
    
    result = minimize(
        fun = neg_log_likelihood_gm,
        x0 = init_params,
        args = (age, Dx, Ex),
        bounds = bounds,
        method = 'L-BFGS-B'
    )
    return result   

def fit_ggm(age, Dx, Ex, bounds, init_params = None, 
            n = 100, meaningless = True, notice = False,
            weight_func = None, weight_params = None, best_neg_logL = None,
            use_rmse_filter = False, rmse_filter_params = None,
            opt_func = "differential_evolution") :
    # 경계 설정
    if bounds is None : bounds = [(0.000005, 0.0005), (0.05, 0.3), (0.07, 0.25), (0.00001, 0.05)]

    epsilon = 1e-7
    best_result = None
    best_neg_logL = best_neg_logL if best_neg_logL is not None else np.inf
    no_improve_count = 0
    boundary_issue_count = 0
    rmse_issue_count = 0
    logL_issue_count = 0

    result_gm = fit_gm(age = age, Dx = Dx, Ex = Ex)
    logL_gm = -result_gm.fun
    
    neg_log_likelihood = make_neg_log_likelihood(Dx = Dx, Ex = Ex, age = age, 
                                                weight_func = weight_func,
                                                weight_params = weight_params)
    
    for i in trange(n, desc = "GGM 적합 진행중") :
            
        try :
            result = run_optimizer(opt_func, neg_log_likelihood, bounds, init_params)    
        except Exception as e :
            print(f"{i + 1}번째 시행 최적화 실패: {e}")   
            continue
        
        params = result.x
        neg_log_likelihood_pure = make_neg_log_likelihood(Dx = Dx, Ex = Ex, age = age, weight_func = None)
        logL_ggm_pure = -neg_log_likelihood_pure(result.x)
        
        if logL_ggm_pure < logL_gm :
            logL_issue_count += 1
            continue

        # 경계에 걸렸는지 확인
        at_boundary = any(
            abs(p - low) < epsilon or abs(p - high) < epsilon
            for p, (low, high) in zip(params, bounds)
        )

        if at_boundary:
            boundary_issue_count += 1
            continue  
        
        if result.fun < best_neg_logL :
            if use_rmse_filter :
                filter_params = rmse_filter_params if rmse_filter_params is not None else {}
                rmse_threshold = filter_params.pop('rmse_threshold', 0.05)
        
            rmse_score = assess_fit_rmse(params, age, Dx, Ex, **filter_params)
            
            if rmse_score > rmse_threshold :
                if notice : rmse_issue_count += 1
                continue    
            
            best_result, best_neg_logL = result , result.fun
            no_improve_count = 0
            if notice :
                print(i + 1, "번째 시도: \n", result.x)  
        else: no_improve_count += 1  
        
        if meaningless and no_improve_count >= 500: 
            print(f"{i + 1}번째에서 500번 연속 개선 없음 → 종료")
            break
    
    if notice :
        print(f"logL issue {logL_issue_count}회 발생")    
        print(f"Boundary issue {boundary_issue_count}회 발생")
        print(f"RMSE issue {rmse_issue_count}회 발생")
    
    return best_result

def fitted_plot(result_ggm, result_gm, mu_obs) :
    fitted_mu_ggm, x_star = calc(result_ggm, age)
    a_gm, b_gm, c_gm = result_gm.x
    fitted_mu_gm = a_gm * np.exp(b_gm * age) + c_gm

    # 시각화
    plt.plot(age, mu_obs, label='Observed', marker='o')
    plt.plot(age, fitted_mu_ggm, label = 'Fitted GGM', linestyle = '--')
    plt.plot(age, fitted_mu_gm, label = 'Fitted GM', linestyle = ':')
    plt.xlabel('Age')
    plt.ylabel('Mortality Rate')
    plt.title('Gamma-Gompertz-Makeham Fit')
    plt.legend()
    plt.grid(True)
    plt.show()


def calc(result, age) :
    a, b, gamma, c = result.x
    log_num = np.log(a) + b * age
    log_denom = np.log1p((gamma * a / b) * (np.expm1(b * age))) 
    fitted_mu = np.exp(log_num - log_denom) + c
    
    num = (b + c * gamma) * c
    denom = 2 * a * b
    root_numer = (b + c * gamma) * c * gamma * ((b + c * gamma) * c - 4 * b * (a * gamma - b))
    root_denom = 2 * a * b * gamma
    log_argument = (num / denom) + (np.sqrt(root_numer) / root_denom)
    
    x_star = (1 / b) * np.log(log_argument)  
    
    return fitted_mu, x_star
    
def run_batch(years, sex, df, output_path, trial = 100, 
            use_weights = True, use_rmse_filter = True,
            center = 80, scale = 3, max_weight = 5, threshold = 0.005,
            opt_func = "differential_evolution") :
    """
    여러 연도와 성별에 대해 GGM과 GM 모델을 적합하고 결과를 CSV로 저장

    Parameters:
        years: iterable of int
        sex: '남자' or '여자' or '전체'
        output_path: 결과 CSV 경로
    """
    records = []
    
    for year in years:
        # 연도별 lx, Ex 불러오기
        print(f"{year}년 시작")
        year, sex, Dx, Ex, age, observed_mu = load_excel(year = year, sex = sex)
        
        weight_func = weight_sigmoid if use_weights else None
        weight_params = {'center' : center, 'scale' : scale, 'max_weight' : max_weight}
        function_params = {'center' : center, 'scale' : scale, 
                            'max_weight' : max_weight, 'rmse_threshold' : threshold}
        
        # GGM 적합
        result = fit_ggm(age, Dx, Ex, n = trial, notice = True, 
                            weight_func = weight_func,
                            weight_params = weight_params,
                            use_rmse_filter = use_rmse_filter, 
                            rmse_filter_params = function_params,
                            opt_func = opt_func)
        
        if result and result.success:
            a, b, gamma, c = result.x
            fitted_mu, x_star = calc(result, age)
            
        # 결과 레코드 생성
        base = {'sex': sex,
                'year': year,
                'a': a, 'b': b, 'gamma': gamma, 'c': c, 
                'x*': x_star}
        
        for idx, age_val in enumerate(age):
            base[f'fitted_ggm_{int(age_val)}'] = fitted_mu[idx]
        records.append(base)
        print(f"{year}년 끝")
            
    df_out = pd.DataFrame.from_records(records)
    df_out.to_csv(output_path, index=False, encoding = 'utf-8-sig')
    print(f"Results saved to {output_path}")        
    
    
def run_test(year, sex, df, trial = 100, use_weights = True, use_rmse_filter = True, notice = True,
            center = 80, scale = 3, max_weight = 5, threshold = 0.005, show_graph = True,
            result_path = None, opt_func = "differential_evolution") :
    records = []
    year, sex, Dx, Ex, age, observed_mu = load_excel(year = year, sex = sex)
    
    weight_func = weight_sigmoid if use_weights else None
    weight_params = {'center' : center, 'scale' : scale, 'max_weight' : max_weight}
    function_params = {'center' : center, 'scale' : scale, 
                        'max_weight' : max_weight, 'rmse_threshold' : threshold}
    
    # GGM 적합
    result_ggm = fit_ggm(age, Dx, Ex, n = trial, notice = notice, bounds = None,
                        weight_func = weight_func, 
                        weight_params = weight_params,
                        use_rmse_filter = use_rmse_filter,
                        rmse_filter_params = function_params,
                        opt_func = opt_func)
    if result_ggm and result_ggm.success:
        a, b, gamma, c = result_ggm.x
        fitted_mu, x_star = calc(result_ggm, age)
        
    if result_path is not None :
        # 결과 레코드 생성
        base = {
            'sex': sex,
            'year': year,
            'a': a, 'b': b, 'gamma': gamma, 'c': c, 
            'x*': x_star}
    
        for idx, age_val in enumerate(age):
            base[f'fitted_ggm_{int(age_val)}'] = fitted_mu[idx]
        records.append(base)
        df_out = pd.DataFrame.from_records(records)
        replace_result_for_year(year = year, sex = sex, new_row = df_out,
                                result_path = result_path)
    
    # weight 없는 순수 logL 계산
    neg_log_likelihood_pure = make_neg_log_likelihood(Dx = Dx, Ex = Ex, age = age, weight_func = None)
    logL_ggm_pure = -neg_log_likelihood_pure(result_ggm.x)
    
    # GM 결과 생성
    result_gm = fit_gm(age = age, Dx = Dx, Ex = Ex)
    a_gm, b_gm, c_gm = result_gm.x

    if notice : 
        print("GGM 로그우도 :", logL_ggm_pure)
        print("GM 로그우도 :", -result_gm.fun)
        
        print("추정 결과:")
        print(f"a     = {a:.15f}")
        print(f"b     = {b:.15f}")
        print(f"gamma = {gamma:.15f}")
        print(f"c     = {c:.15f}")
        
        print("추정된 감속 나이 x* =", round(x_star, 2), "세")

        print(f"a_gm     = {a_gm:.15f}")
        print(f"b_gm     = {b_gm:.15f}")
        print(f"c_gm     = {c_gm:.15f}")
        
    
    if show_graph :
        fitted_plot(result_ggm, result_gm, observed_mu)
    
    return result_ggm
    
def replace_result_for_year(year, sex, new_row, result_path):
    """
    기존 결과 파일에서 특정 연도(year), 성별(sex)의 결과만 새로 교체함

    Parameters:
        year: int
        sex: '남자' or '여자'
        new_row: dict 형태의 새 결과 (기존 run_batch에서 생성한 base와 동일 포맷)
        result_path: CSV 파일 경로
    """
    try:
        df_all = pd.read_csv(result_path)
    except FileNotFoundError:
        print("[경고] 기존 결과 파일이 없습니다. 새로 생성합니다.")
        df_all = pd.DataFrame()

    # 해당 연도·성별 기존 행은 제거
    if not df_all.empty:
        df_all = df_all[~((df_all['sex'] == sex) & (df_all['year'] == year))]
    # new_row가 DataFrame인지 dict/Series인지 판별하여 concat
    if isinstance(new_row, pd.DataFrame):
        df_to_add = new_row.copy()
    elif isinstance(new_row, dict):
        df_to_add = pd.DataFrame([new_row])
    elif isinstance(new_row, pd.Series):
        # 하나의 Series라면 dict로 변환
        df_to_add = pd.DataFrame([new_row.to_dict()])
    else:
        raise ValueError(f"Unsupported type for new_row: {type(new_row)}. "
                        "Expect dict, Series, or DataFrame.")
    # 병합: 기존 df_all과 새로운 df_to_add를 합침
    df_all = pd.concat([df_all, df_to_add], ignore_index=True)
    # 정렬
    if 'sex' in df_all.columns and 'year' in df_all.columns:
        df_all = df_all.sort_values(['sex', 'year']).reset_index(drop=True)
    # 저장
    df_all.to_csv(result_path, index=False, encoding='utf-8-sig')
    print(f"{year}년 {sex} 결과가 갱신되었습니다 → {result_path}")
    
    
def draw_LAR (params, age):
    a, b, gamma, c = params
    log_num = np.log(a) + b * age
    log_denom = np.log1p((gamma * a / b) * (np.expm1(b * age))) 
    mu = np.exp(log_num - log_denom) + c
    lar = b * (1 - c / mu) - gamma * (1 - c / mu) * (mu - c)
    
    num = (b + c * gamma) * c
    denom = 2 * a * b
    root_numer = (b + c * gamma) * c * gamma * ((b + c * gamma) * c - 4 * b * (a * gamma - b))
    root_denom = 2 * a * b * gamma
    log_argument = (num / denom) + (np.sqrt(root_numer) / root_denom)
    
    x_star = (1 / b) * np.log(log_argument)  
    print("x* : ", x_star, "세")
    
    plt.plot(age, lar, label='Fitted', linestyle='--')
    plt.xlabel('Age')
    plt.ylabel('LAR')
    plt.title('Gamma-Gompertz-Makeham Fit')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def find_best_scale (year, sex, trial,
                    center_range, scale_range, max_weight_range, best_logL = None,
                    n_runs = 30, Dx = None, Ex = None, age = None) :
    
    best_logL = best_logL if best_logL is not None else -np.inf
    best_result = None; best_scale_params = None
    improve_count = 0

    if isinstance(center_range, (tuple, list)):
        center_candidates = list(range(center_range[0], center_range[1] + 1, center_range[2]))
    else:
        center_candidates = [center_range]

    if isinstance(scale_range, (tuple, list)):
        scale_candidates = [round(x, 1) for x in np.arange(*scale_range)]
    else:
        scale_candidates = [scale_range]

    if isinstance(max_weight_range, (tuple, list)):
        max_weight_candidates = list(range(max_weight_range[0], max_weight_range[1] + 1, max_weight_range[2]))
    else:
        max_weight_candidates = [max_weight_range]

    neg_log_likelihood_pure = make_neg_log_likelihood(Dx = Dx, Ex = Ex, age = age, weight_func = None)
    for i in trange(n_runs, desc = f"Searching best scale for {year} {sex}"):
        center = random.choice(center_candidates)
        scale = random.choice(scale_candidates)
        max_weight = random.choice(max_weight_candidates)
        
        try:
            result_ggm = run_test(year = year, sex = sex, df = df, trial = trial, notice = False,
                                center = center, scale = scale, max_weight = max_weight, show_graph = False)
        except Exception as e:
            print(f"실패: {e}")
            continue
        
        logL_ggm_pure = -neg_log_likelihood_pure(result_ggm.x)

        if logL_ggm_pure > best_logL :
            improve_count += 1
            best_logL = logL_ggm_pure
            best_scale_params = {
                "center": center,
                "scale": scale,
                "max_weight": max_weight
            }
            best_result = result_ggm
            a_best, b_best, gamma_best, c_best = best_result.x             

    if improve_count == 0 : print("개선 실패")
    elif improve_count != 0 :
        print(improve_count, "회 개선 성공")
        print(f"최고 로그우도 : {best_logL}")
        print("최적 scale:")
        print(f"center     = {best_scale_params['center']}")
        print(f"scale      = {best_scale_params['scale']}")
        print(f"max_weight = {best_scale_params['max_weight']}")
        print("---------------------------")
        print(f"a     = {a_best:.10f}")
        print(f"b     = {b_best:.10f}")
        print(f"gamma = {gamma_best:.10f}")
        print(f"c     = {c_best:.10f}")
    
    return best_result, best_logL, best_scale_params

# TODO 단순 logL뿐만 아니라 AICc같은 것도 비교해야 할듯... 근데 뭘 써야할지 잘 모르겠음

def save_scale_result_to_excel(result, logL_ggm_pure, best_scale_params, year, sex, filepath):
    """
    최적 GGM 파라미터 및 가중치 파라미터를 엑셀로 저장

    A1: 연도
    A2: 성별
    A3~: a, b, gamma, c, logL, center, scale, max_weight
    """
    a, b, gamma, c = result.x
    center = best_scale_params['center']
    scale = best_scale_params['scale']
    max_weight = best_scale_params['max_weight']

    # 저장할 데이터프레임 구성
    new_data = pd.DataFrame([{
        "year": year,
        "sex": sex,
        "a": a,
        "b": b,
        "gamma": gamma,
        "c": c,
        "logL": logL_ggm_pure,
        "center": center,
        "scale": scale,
        "max_weight": max_weight
    }])

    if os.path.exists(filepath):
        try:
            existing_data = pd.read_excel(filepath)
        except Exception as e:
            print(f"기존 파일 읽기 실패: {e}")
            existing_data = pd.DataFrame()
    else:
        existing_data = pd.DataFrame()

    # 기존에 동일한 year & sex가 있다면 삭제 후 새로 추가
    if not existing_data.empty:
        mask = (existing_data['year'] == year) & (existing_data['sex'] == sex)
        existing_data = existing_data[~mask]

    # 데이터 합치기
    updated_data = pd.concat([existing_data, new_data], ignore_index=True)

    # 저장
    updated_data.to_excel(filepath, index=False)    

def get_scale_data_from_file(filepath, year, sex, default_center = None, default_scale = None, default_max_weight = None):
    """
    결과 파일에서 (year, sex)에 해당하는 logL, center, scale, max_weight을 불러옵니다.
    해당 데이터가 없으면 None 반환
    """
    if not os.path.exists(filepath):
        print(f"파일이 존재하지 않습니다: {filepath}")
        return {
            'logL': None,
            'center': default_center,
            'scale': default_scale,
            'max_weight': default_max_weight
        }
    
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"파일 읽기 실패: {e}")
        return {
            'logL': None,
            'center': default_center,
            'scale': default_scale,
            'max_weight': default_max_weight
        }

    mask = (df['year'] == year) & (df['sex'] == sex)
    matched = df[mask]

    if matched.empty:
        return {
            'logL': None,
            'center': default_center,
            'scale': default_scale,
            'max_weight': default_max_weight
        }
    row = matched.iloc[0]

    return {
        'logL': row.get('logL', None),
        'center': row.get('center', default_center),
        'scale': row.get('scale', default_scale),
        'max_weight': row.get('max_weight', default_max_weight)
    }