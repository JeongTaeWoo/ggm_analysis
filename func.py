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

def load_life_table(year, sex, df = df) :
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
def fit_gm(age, Dx, Ex, bounds=[(0, 0.0005), (0.01, 0.5), (0.000001, 0.05)]):
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

def fit_ggm(age, Dx, Ex, bounds = None, init_params = None, best_logL_gm = None,
            n = 100, meaningless = True, notice = False,
            weight_func = None, weight_params = None, best_logL_ggm = None,
            use_rmse_filter = True, rmse_filter_params = None,
            opt_func = "differential_evolution") :
    # 경계 설정
    if bounds is None : bounds = [(0.000005, 0.0005), (0.05, 0.2), (0.07, 0.2), (0.00001, 0.05)]
    
    best_logL_gm = best_logL_gm if best_logL_gm is not None else -np.inf
    best_logL_ggm = best_logL_ggm if best_logL_ggm is not None else -np.inf
    epsilon = 1e-7
    best_result = None

    no_improve_count = 0
    boundary_issue_count = 0
    rmse_issue_count = 0
    logL_issue_count = 0
    
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
        
        if logL_ggm_pure < best_logL_gm :
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
        
        if logL_ggm_pure > best_logL_ggm :
            if use_rmse_filter :
                filter_params = rmse_filter_params if rmse_filter_params is not None else {}
                rmse_threshold = filter_params.pop('rmse_threshold', 0.05)
        
            rmse_score = assess_fit_rmse(params, age, Dx, Ex, **filter_params)
            
            if rmse_score > rmse_threshold :
                if notice : rmse_issue_count += 1
                continue    

            best_result = result
            best_logL_ggm = logL_ggm_pure # best logL 업데이트
            
            no_improve_count = 0
            if notice :
                print(f"{i + 1} 번째 시도:  {result.x}")  
        else: no_improve_count += 1  
        
        if meaningless and no_improve_count >= 500: 
            print(f"{i + 1}번째에서 500번 연속 개선 없음 → 종료")
            break
    
    if notice :
        print(f"logL issue {logL_issue_count}회 발생")    
        print(f"Boundary issue {boundary_issue_count}회 발생")
        print(f"RMSE issue {rmse_issue_count}회 발생")
    
    return best_result

def draw_fitted_plot(ggm_params, gm_params, mu_obs, age):
    a_gm, b_gm, c_gm = gm_params
    fitted_mu_ggm, _ = calc_ggm(ggm_params, age)
    fitted_mu_gm = a_gm * np.exp(b_gm * age) + c_gm

    plt.plot(age, mu_obs, label='Observed', marker='o')
    plt.plot(age, fitted_mu_ggm, label='Fitted GGM', linestyle='--')
    plt.plot(age, fitted_mu_gm, label='Fitted GM', linestyle=':')
    plt.xlabel('Age')
    plt.ylabel('Mortality Rate')
    plt.title('Gamma-Gompertz-Makeham Fit')
    plt.legend()
    plt.grid(True)
    plt.show()


def fitting_gm(year, sex, age, show_graph = True):
    year, sex, Dx, Ex, age, observed_mu = load_life_table(year = year, sex = sex)
    result = fit_gm(age = age, Dx = Dx, Ex = Ex)
    a, b, c = result.x
    fitted_mu_gm = a * np.exp(b * age) + c

    if show_graph : 
        print(f"a_gm     = {a:.15f}")
        print(f"b_gm     = {b:.15f}")
        print(f"c_gm     = {c:.15f}")
        plt.plot(age, observed_mu, label='Observed', marker='o')
        plt.plot(age, fitted_mu_gm, label = 'Fitted GM', linestyle = ':')
        plt.xlabel('Age')
        plt.ylabel('Mortality Rate')
        plt.title('Gompertz-Makeham Fit')
        plt.legend()
        plt.grid(True)
        plt.show()

    return result

def calc_ggm(params, age):
    a, b, gamma, c = params
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
    
    
def run_test(year, sex, df, trial = 100, use_weights = True, use_rmse_filter = True, notice_bool = True,
            center = 90, scale = 3, max_weight = 5, threshold = 0.005, opt_func = "differential_evolution"):
    year, sex, Dx, Ex, age, observed_mu = load_life_table(year = year, sex = sex)

    weight_func = weight_sigmoid if use_weights else None
    weight_params = {'center': center, 'scale': scale, 'max_weight': max_weight}
    function_params = {'center': center, 'scale': scale, 
                        'max_weight': max_weight, 'rmse_threshold': threshold}

    # GGM 적합
    result_ggm = fit_ggm(age, Dx, Ex, n = trial, notice = notice_bool, bounds = None,
                        weight_func = weight_func, 
                        weight_params = weight_params,
                        use_rmse_filter = use_rmse_filter,
                        rmse_filter_params = function_params,
                        opt_func = opt_func)

    if result_ggm and result_ggm.success:
        a, b, gamma, c = result_ggm.x
        fitted_mu, x_star = calc_ggm(result_ggm.x, age)
        logL_ggm_pure = -make_neg_log_likelihood(Dx, Ex, age, weight_func=None)(result_ggm.x)
    else:
        if fallback_filepath:
            scale_row = get_data_from_file(fallback_filepath, year, sex)
            a, b, gamma, c = scale_row['a'], scale_row['b'], scale_row['gamma'], scale_row['c']
            fitted_mu, x_star = calc_ggm([a, b, gamma, c], age)
            result_ggm = result_maker(a, b, gamma, c)
            logL_ggm_pure = scale_row['logL_ggm'] if scale_row['logL_ggm'] is not None else -np.inf
            if notice:
                print("GGM 적합 실패 → 기존 파라미터로 대체")
        else:
            raise RuntimeError("GGM 적합 실패 및 fallback 파일 없음")

    # 기존보다 개선된 경우만 저장할 데이터 구성
    best_result = result_ggm
    best_logL = logL_ggm_pure
    best_scale_params = {'center': center, 'scale': scale, 'max_weight': max_weight}

    return best_result, best_logL, best_scale_params
    
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
    
def find_best_scale (year, sex, trial, center_range, scale_range, max_weight_range, filepath,
                    best_logL_ggm = None, best_logL_gm = None, n_runs = 30, Dx = None, Ex = None, age = None, show_graph = True, threshold = 0.005) :
    
    result_gm = fitting_gm(year = year, sex = sex, age = age, show_graph = show_graph) 
    best_logL_gm = best_logL_gm if best_logL_gm is not None else -np.inf
    best_logL_ggm = best_logL_ggm if best_logL_ggm is not None else -np.inf
    best_result = None; best_scale_params = None
    improve_count = 0
    fitting_fail_count = 0
    year, sex, Dx, Ex, age, observed_mu = load_life_table(year = year, sex = sex)

    if -result_gm.fun > best_logL_gm :
        best_logL_gm = -result_gm.fun
        print("GM 개선 성공")
    else: 
        print("GM 개선 실패")

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
            result_ggm = fit_ggm(age, Dx, Ex, n = trial, notice = False, bounds = None, best_logL_gm = best_logL_gm,
                            weight_func = weight_sigmoid,
                            weight_params = {'center': center, 'scale': scale, 'max_weight': max_weight},
                            rmse_filter_params = {'center': center, 'scale': scale, 'max_weight': max_weight, 'rmse_threshold': threshold},
                            opt_func = "differential_evolution")
        except Exception as e:
            print(f"적합 실패: {e}")
            fitting_fail_count += 1
            continue
        
        logL_ggm_pure = -neg_log_likelihood_pure(result_ggm.x)

        if logL_ggm_pure > best_logL_ggm :
            improve_count += 1
            best_logL_ggm = logL_ggm_pure
            best_scale_params = { "center": center, "scale": scale, "max_weight": max_weight}
            best_result = result_ggm
            a_best, b_best, gamma_best, c_best = best_result.x        

    update_values = None
    if improve_count > 0 :
        a, b, gamma, c = best_result.x
        center = best_scale_params['center']
        scale = best_scale_params['scale']
        max_weight = best_scale_params['max_weight']
        fitted_mu, x_star = calc_ggm(best_result.x, age)
        a_gm, b_gm, c_gm = result_gm.x
        update_values = {
            "a": a, "b": b, "gamma": gamma, "c": c,
            "logL_ggm": best_logL_ggm, "center": center, "scale": scale, "max_weight": max_weight, "x*": x_star,
            "a_gm": a_gm, "b_gm": b_gm, "c_gm": c_gm, "logL_gm": -result_gm.fun
        }

    elif -result_gm.fun > best_logL_gm : 
        a_gm, b_gm, c_gm = result_gm.x
        update_values = {
            "a_gm": a_gm, "b_gm": b_gm, "c_gm": c_gm, "logL_gm": -result_gm.fun
        }
    else:
        if filepath is None:
            raise ValueError("개선 실패 시 기존 파라미터를 불러오기 위해 filepath 인자가 필요합니다.")
        scale_row = get_data_from_file(filepath, year, sex)
        ggm_params = [scale_row['a'], scale_row['b'], scale_row['gamma'], scale_row['c']]

    if update_values : # 결과에 맞게 유동적으로 엑셀에 저장
        save_result_to_excel(update_values, year, sex, filepath)


    if show_graph :
        draw_fitted_plot(ggm_params, result_gm.x, observed_mu, age)

    if improve_count > 0:
        print(improve_count, "회 개선 성공")
        print(f"최고 로그우도 : {best_logL_ggm}")
        print("최적 scale:")
        print(f"center     = {best_scale_params['center']}")
        print(f"scale      = {best_scale_params['scale']}")
        print(f"max_weight = {best_scale_params['max_weight']}")
        print("---------------------------")
        a_best, b_best, gamma_best, c_best = best_result.x
        print(f"a     = {a_best:.10f}")
        print(f"b     = {b_best:.10f}")
        print(f"gamma = {gamma_best:.10f}")
        print(f"c     = {c_best:.10f}")

    else:
        print("개선 실패: 기존 GGM 파라미터로 그래프만 출력했습니다.")


    return best_result, best_logL_ggm, best_scale_params, result_gm


def save_result_to_excel(update_values: dict, year, sex, filepath):

    # 기존 데이터 불러오기
    if os.path.exists(filepath):
        try:
            existing_data = pd.read_excel(filepath)
        except Exception as e:
            print(f"기존 파일 읽기 실패: {e}")
            existing_data = pd.DataFrame()
    else:
        existing_data = pd.DataFrame()

    mask = (existing_data['year'] == year) & (existing_data['sex'] == sex)

    if mask.any():
        # 기존 행이 있으면 해당 컬럼만 업데이트
        for key, value in update_values.items():
            existing_data.loc[mask, key] = value
        updated_data = existing_data
    else:
        # 없으면 새 행 추가
        new_row = {'year': year, 'sex': sex}
        new_row.update(update_values)
        updated_data = pd.concat([existing_data, pd.DataFrame([new_row])], ignore_index=True)

    # 저장
    updated_data.to_excel(filepath, index=False)


def get_data_from_file(filepath, year, sex, default_value = None):
    """
    결과 파일에서 (year, sex)에 해당하는 데이터를 딕셔너리로 반환합니다.
    - 해당 데이터가 없으면 default_values를 기반으로 생성된 dict를 반환합니다.
    - default_values는 {'center': x, 'scale': y, ...} 형태로 전달 가능
    """
    if default_value is None:
        default_value = {}

    if not os.path.exists(filepath):
        print(f"파일이 존재하지 않습니다: {filepath}")
        return {'logL_ggm': None, 'logL_gm': None, **default_value}
    
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"파일 읽기 실패: {e}")
        return {'logL_ggm': None, 'logL_gm': None, **default_value}
    
    mask = (df['year'] == year) & (df['sex'] == sex)
    matched = df[mask]

    if matched.empty:
        return {'logL_ggm': None, 'logL_gm': None, **default_value}

    row = matched.iloc[0].to_dict()
    row = {k: (None if pd.isna(v) else v) for k, v in row.items()}

    # 필수 항목들 None 반환
    essential_keys = ['logL_ggm', 'logL_gm']
    for key in essential_keys:
        if key not in row:
            row[key] = None

    # default_values에 있는 값으로 결측 채우기
    for key, default in default_value.items():
        if pd.isna(row.get(key)):
            row[key] = default
    
    return row