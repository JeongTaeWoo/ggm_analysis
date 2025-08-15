from pathlib import Path
import random
from scipy.optimize import differential_evolution, minimize, dual_annealing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import trange
import traceback

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
def fit_gm(age, Dx, Ex, bounds=[(1e-7, 1e-3), (0.01, 0.3), (1e-6, 1e-3)]):
    init_params = [1e-5, 0.1, 1e-4]
    
    result = minimize(
        fun = neg_log_likelihood_gm,
        x0 = init_params,
        args = (age, Dx, Ex),
        bounds = bounds,
        method = 'L-BFGS-B'
    )
    return result   

def fit_ggm(age, Dx, Ex, sex, bounds = None, init_params = None, best_logL_gm = None,
            n = 100, meaningless = True, notice_issue = False, notice_trange = False,
            weight_func = None, weight_params = None, best_logL_ggm = None, compare_gm = False,
            opt_func = "differential_evolution") :
    # 경계 설정
    if bounds is None : 
        if sex == '남자' :
            bounds = [(1e-4, 3e-3), (0.08, 0.14), (0.03, 0.3), (3e-5, 3e-3)]
        elif sex == '여자' :
            bounds = [(2e-5, 1e-3), (0.08, 0.15), (0.03, 0.3), (3e-4, 3e-3)]    
    
    best_logL_gm = best_logL_gm if best_logL_gm is not None else -np.inf
    best_logL_ggm = best_logL_ggm if best_logL_ggm is not None else -np.inf
    epsilon = 1e-7
    best_result = None
    best_logL_ggm_temp = -np.inf

    no_improve_count = 0
    boundary_issue_count = 0
    logL_issue_count = 0
    
    neg_log_likelihood = make_neg_log_likelihood(Dx, Ex, age, weight_func, weight_params)
    
    for i in trange(n, desc = "GGM 적합 진행중", disable = not notice_trange) :
        try :
            result = run_optimizer(opt_func, neg_log_likelihood, bounds, init_params)    
        except Exception as e :
            print(f"{i + 1}번째 시행 최적화 실패: {e}")   
            continue
        
        params = result.x
        neg_log_likelihood_pure = make_neg_log_likelihood(Dx = Dx, Ex = Ex, age = age, weight_func = None)
        logL_ggm_pure = -neg_log_likelihood_pure(result.x)
        
        if logL_ggm_pure > best_logL_ggm_temp :
            best_logL_ggm_temp = logL_ggm_pure

        if compare_gm and logL_ggm_pure < best_logL_gm :
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
            best_result = result
            best_logL_ggm = logL_ggm_pure # best logL 업데이트
            
            no_improve_count = 0
            if notice_issue :
                print(f"{i + 1} 번째 시도:  {result.x}")  
        else: no_improve_count += 1  
        
        if meaningless and no_improve_count >= 500: 
            print(f"{i + 1}번째에서 500번 연속 개선 없음 → 종료")
            break
    
    if notice_issue :
        print(f"logL issue {logL_issue_count}회 발생")    
        print(f"Boundary issue {boundary_issue_count}회 발생")
    
    return best_result, best_logL_ggm_temp

def draw_fitted_plot(ggm_params, gm_params, mu_obs, age, year, sex):
    a_gm, b_gm, c_gm = gm_params
    fitted_mu_ggm, _ = calc_ggm(ggm_params, age)
    fitted_mu_gm = a_gm * np.exp(b_gm * age) + c_gm

    plt.plot(age, mu_obs, label='Observed', marker='o')
    plt.plot(age, fitted_mu_ggm, label='Fitted GGM', linestyle='--')
    plt.plot(age, fitted_mu_gm, label='Fitted GM', linestyle=':')
    plt.xlabel('Age')
    plt.ylabel('Mortality Rate')
    if year or sex is None :
        plt.title('GGM Fit Result')
    elif year and sex is not None :
        if sex == '남자' :
            plt.title(f'GGM Fit Result ({year}, Male)')
        elif sex == '여자' :
            plt.title(f'GGM Fit Result ({year}, Female)')
    plt.legend()
    plt.grid(True)
    plt.show()


def draw_fitted_gm(year, sex, age, show_graph = True):
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
    

def evaluate_fit_metrics(observed_mu, fitted_mu, precision = 6, notice = True) :
    observed_mu = np.array(observed_mu)
    fitted_mu = np.array(fitted_mu)

    rmse = np.sqrt(np.mean((observed_mu - fitted_mu) ** 2))
    mae = np.mean(np.abs(observed_mu - fitted_mu))
    mape = np.mean(np.abs((observed_mu - fitted_mu) / observed_mu)) * 100  # 퍼센트 오차

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }

    if notice:
        print("\n[적합 품질 평가]")
        # RMSE 평가
        if rmse <= 0.01: print(f"RMSE: {metrics['rmse']} → 매우 좋음")
        elif rmse <= 0.03: print(f"RMSE: {metrics['rmse']} → 무난")
        else: print(f"RMSE: {metrics['rmse']} → 별로")
        # MAE
        if mae <= 0.01: print(f"MAE: {metrics['mae']} → 매우 좋음")
        elif mae <= 0.03: print(f"MAE: {metrics['mae']} → 무난")
        else: print(f"MAE: {metrics['mae']} → 별로")
        # MAPE 평가
        if mape <= 5: print(f"MAPE: {metrics['mape']}% → 매우 좋음")
        elif mape <= 10: print(f"MAPE: {metrics['mape']}% → 무난")
        else: print(f"MAPE: {metrics['mape']}% → 별로")
        print("-----------------------------")

    return {k: round(v, precision) for k, v in metrics.items()}


def find_best_scale (year, sex, Dx, Ex, age, trial, center_range, scale_range, max_weight_range, filepath, 
                    notice = True, compare_gm = False, best_logL_ggm = None, best_logL_gm = None, n_runs = 30, 
                    show_graph = True, threshold = 0.005, bounds = None) :
    
    result_gm = draw_fitted_gm(year = year, sex = sex, age = age, show_graph = show_graph) 
    best_logL_gm = best_logL_gm if best_logL_gm is not None else -np.inf
    best_logL_ggm = best_logL_ggm if best_logL_ggm is not None else -np.inf
    best_result = None; best_scale_params = None
    improve_count = 0
    fitting_fail_count = 0
    gm_improve_bool = False
    update_values = None
    required_keys = ['a', 'b', 'gamma', 'c']
    temp_best_logL_ggm = -np.inf 
    temp_best_scale_params = None
    temp_best_logL_ggm_fit = -np.inf
    temp_best_result_fit = None


    year, sex, Dx, Ex, age, observed_mu = load_life_table(year = year, sex = sex)

    if -result_gm.fun > best_logL_gm :
        best_logL_gm = -result_gm.fun
        gm_improve_bool = True
        if notice: print("GM 개선 성공")
    else: 
        if notice: print("GM 개선 실패")

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
    for i in trange(n_runs, desc = f"Searching best scale for {year} {sex}", disable = not notice):
        center = random.choice(center_candidates)
        scale = random.choice(scale_candidates)
        max_weight = random.choice(max_weight_candidates)
        
        try:
            result_ggm, best_logL_ggm_fit = fit_ggm(age, Dx, Ex, sex, n = trial, bounds = bounds, best_logL_gm = best_logL_gm,
                            weight_func = weight_sigmoid, meaningless = False, notice_trange = notice,
                            weight_params = {'center': center, 'scale': scale, 'max_weight': max_weight}, compare_gm = compare_gm,
                            opt_func = "differential_evolution")
            if best_logL_ggm_fit is -np.inf : # 기존 결과보다 안좋은 결과를 얻었다는 뜻
                temp_best_logL_ggm_fit = best_logL_ggm_fit
                temp_best_result_fit = result_ggm
                fitting_fail_count += 1
                continue
        except Exception as e:
            print(f"적합 중 오류 발생: {e}")
            traceback.print_exc()
            fitting_fail_count += 1
            continue
        
        logL_ggm_pure = -neg_log_likelihood_pure(result_ggm.x)

        if logL_ggm_pure > best_logL_ggm :
            improve_count += 1
            best_logL_ggm = logL_ggm_pure
            best_scale_params = { "center": center, "scale": scale, "max_weight": max_weight}
            best_result = result_ggm
            a_best, b_best, gamma_best, c_best = best_result.x 
        
        if logL_ggm_pure > temp_best_logL_ggm : # 기존 결과 개선 실패했더라도 이번에 가장 좋았던 결과 뭐였는지 확인용
            temp_best_logL_ggm = logL_ggm_pure     
            temp_best_scale_params = { "center": center, "scale": scale, "max_weight": max_weight}      

    if improve_count > 0 and gm_improve_bool == True: 
        a, b, gamma, c = best_result.x
        ggm_params = [a, b, gamma, c]
        center = best_scale_params['center']
        scale = best_scale_params['scale']
        max_weight = best_scale_params['max_weight']
        fitted_mu, x_star = calc_ggm(best_result.x, age)
        a_gm, b_gm, c_gm = result_gm.x

        metrics = evaluate_fit_metrics(observed_mu, fitted_mu, notice = notice)
        update_values = {
            "a": a, "b": b, "gamma": gamma, "c": c,
            "logL_ggm": best_logL_ggm, "center": center, "scale": scale, "max_weight": max_weight, "x*": x_star,
            "a_gm": a_gm, "b_gm": b_gm, "c_gm": c_gm, "logL_gm": -result_gm.fun, **metrics
        }
    elif improve_count > 0 and gm_improve_bool == False :
        a, b, gamma, c = best_result.x
        ggm_params = [a, b, gamma, c]
        center = best_scale_params['center']
        scale = best_scale_params['scale']
        max_weight = best_scale_params['max_weight']
        fitted_mu, x_star = calc_ggm(best_result.x, age)

        metrics = evaluate_fit_metrics(observed_mu, fitted_mu, notice = notice)
        update_values = {
            "a": a, "b": b, "gamma": gamma, "c": c,
            "logL_ggm": best_logL_ggm, "center": center, "scale": scale, "max_weight": max_weight, "x*": x_star, **metrics
        }    
    elif improve_count == 0 and gm_improve_bool == True : 
        scale_row = get_data_from_file(filepath, year, sex)
        if scale_row is not None and all((k in scale_row) and (scale_row[k]) is not None for k in required_keys):
            ggm_params = [scale_row['a'], scale_row['b'], scale_row['gamma'], scale_row['c']]  
        else: 
            ggm_params = None    
        a_gm, b_gm, c_gm = result_gm.x
        update_values = {
            "a_gm": a_gm, "b_gm": b_gm, "c_gm": c_gm, "logL_gm": -result_gm.fun
        }
    else:
        if filepath is None:
            raise ValueError("filepath error")
        scale_row = get_data_from_file(filepath, year, sex)
        if scale_row is not None and all(k in scale_row and scale_row[k] is not None for k in required_keys):
            ggm_params = [scale_row['a'], scale_row['b'], scale_row['gamma'], scale_row['c']]  
        else: 
            ggm_params = None

    if update_values is not None : # 결과에 맞게 유동적으로 엑셀에 저장
        save_result_to_excel(update_values, year, sex, filepath)
        if notice: print("결과 저장 성공")
    else : 
        if notice: print("결과 개선 실패")

    if show_graph and ggm_params is not None:
        draw_fitted_plot(ggm_params, result_gm.x, observed_mu, age, year, sex)    

    if notice :
        if improve_count > 0:
            print(improve_count, "회 개선 성공")
            print(f"최고 로그우도 : {best_logL_ggm}")
            print("최적 scale:")
            print(f"center     = {best_scale_params['center']}")
            print(f"scale      = {best_scale_params['scale']}")
            print(f"max_weight = {best_scale_params['max_weight']}")
            print(f"x* = {x_star:2f}세")
            print("---------------------------")
            a_best, b_best, gamma_best, c_best = best_result.x
            print(f"a     = {a_best:.10f}")
            print(f"b     = {b_best:.10f}")
            print(f"gamma = {gamma_best:.10f}")
            print(f"c     = {c_best:.10f}")
        elif improve_count == 0 and fitting_fail_count != n_runs:
            print(f"이번 시행의 최고 로그우도와 기존 값의 차이 : {best_logL_ggm - temp_best_logL_ggm}")
            print("이번 시행의 최적 scale:")
            print(f"center     = {temp_best_scale_params['center']}")
            print(f"scale      = {temp_best_scale_params['scale']}")
            print(f"max_weight = {temp_best_scale_params['max_weight']}")
            if temp_best_result_fit is not None:    
                a_temp, b_temp, gamma_temp, c_temp = temp_best_result_fit.x
                print(f"a     = {a_temp:.10f}")
                print(f"b     = {b_temp:.10f}")
                print(f"gamma = {gamma_temp:.10f}")
                print(f"c     = {c_temp:.10f}")
            if scale_row['x*'] is not None : 
                print(f"x* = {scale_row['x*']:.2f}세")
            if ggm_params is not None :
                print("개선 실패: 기존 GGM 파라미터로 그래프만 출력했습니다.")
            elif ggm_params is None : 
                ("개선 실패: 기존 GGM 파라미터가 없으므로 그래프 생략")     
        
        elif improve_count == 0 and fitting_fail_count == n_runs : 
            if temp_best_result_fit is not None:    
                a_temp, b_temp, gamma_temp, c_temp = temp_best_result_fit.x
                print(f"a     = {a_temp:.10f}")
                print(f"b     = {b_temp:.10f}")
                print(f"gamma = {gamma_temp:.10f}")
                print(f"c     = {c_temp:.10f}")
                ggm_params = [a_temp, b_temp, gamma_temp, c_temp]
                draw_fitted_plot(ggm_params, result_gm.x, observed_mu, age, year, sex)
            # print("GM보다 나은 결과를 한 번도 얻지 못했음")
            # print(f"이번 시행의 최고 로그우도와 GM 값의 차이 : {best_logL_gm - temp_best_logL_ggm_worse}")
            


    return best_result, best_logL_ggm, best_scale_params, result_gm

def _parse_input_to_list(arg):
    """
    None, 단일 값, 또는 range/list를 리스트로 변환하는 헬퍼 함수
    """
    if arg is None:
        return [None]
    elif isinstance(arg, (int, float)):
        return [arg]
    elif isinstance(arg, range):
        return list(arg)
    elif isinstance(arg, (list, tuple, np.ndarray)):
        return list(arg)
    else:
        raise ValueError(f"지원하지 않는 인자 형식: {type(arg)}")
    
def run_refine_search(year, sex, filepath, centers, scales, max_weights, bounds = None):
    """
    주어진 범위의 가중치 파라미터(centers, scales, max_weights) 조합을
    모두 탐색하고 결과를 별도 엑셀 파일에 저장합니다.
    """
    print(f"[{year}년 {sex} 파라미터 미세조정 탐색 시작]")

    # 생명표 데이터 불러오기
    year, sex, Dx, Ex, age, observed_mu = load_life_table(year, sex)

    # 기존 GGM 로그우도 불러오기 (탐색 결과 비교용)
    scale_row = get_data_from_file(filepath, year, sex)
    baseline_logL_ggm = scale_row.get('logL_ggm', -np.inf)
    if baseline_logL_ggm == -np.inf:
        print("기존 GGM 로그우도 값이 없습니다.")
    else:
        print(f"기존 GGM 로그우도: {baseline_logL_ggm:.4f}")

    # 탐색할 파라미터 리스트 생성
    center_list = _parse_input_to_list(centers)
    scale_list = _parse_input_to_list(scales)
    max_weight_list = _parse_input_to_list(max_weights)

    all_results = []
    
    # 순수 로그우도 계산 함수 (가중치 없는)
    neg_log_likelihood_pure = make_neg_log_likelihood(Dx, Ex, age, weight_func=None)

    # 탐색 루프
    total_runs = len(center_list) * len(scale_list) * len(max_weight_list)
    search_bar = trange(total_runs, desc="탐색 진행 중", leave=True)
    
    # 가장 좋았던 결과 저장용 변수
    best_logL_ggm_overall = -np.inf
    best_result_dict = None
    
    for center in center_list:
        for scale in scale_list:
            for max_weight in max_weight_list:
                search_bar.update(1)
                
                # 기존 GGM 파라미터를 초기값으로 사용
                init_params = [scale_row.get(p) for p in ['a', 'b', 'gamma', 'c']]
                if None in init_params:
                    print("오류: 초기 GGM 파라미터가 엑셀에 없어 탐색을 건너뜁니다.")
                    continue
                
                # 가중치 파라미터 설정
                current_weight_params = {
                    'center': center if center is not None else scale_row['center'],
                    'scale': scale if scale is not None else scale_row['scale'],
                    'max_weight': max_weight if max_weight is not None else scale_row['max_weight']
                }

                try:
                    neg_log_likelihood = make_neg_log_likelihood(Dx, Ex, age, weight_func=weight_sigmoid, weight_params = current_weight_params)
                    bounds = bounds if bounds is not None else [(1e-4, 3e-3), (0.08, 0.14), (0.01, 0.3), (3e-5, 3e-3)]
                    result = minimize(
                        fun=neg_log_likelihood,
                        x0=init_params,
                        bounds=bounds,
                        method='L-BFGS-B'
                    )

                    if result.success:
                        new_logL_ggm = -neg_log_likelihood_pure(result.x)
                        fitted_mu, x_star = calc_ggm(result.x, age)
                        metrics = evaluate_fit_metrics(observed_mu, fitted_mu, notice=False)
                        
                        result_dict = {
                            "year": year, "sex": sex,
                            "center": current_weight_params['center'],
                            "scale": current_weight_params['scale'],
                            "max_weight": current_weight_params['max_weight'],
                            "a": result.x[0], "b": result.x[1], "gamma": result.x[2], "c": result.x[3],
                            "logL_ggm": new_logL_ggm,
                            "logL_diff": new_logL_ggm - baseline_logL_ggm,
                            "x*": x_star,
                            **metrics
                        }
                        all_results.append(result_dict)
                        
                        # 가장 좋았던 결과 갱신
                        if new_logL_ggm > best_logL_ggm_overall:
                            best_logL_ggm_overall = new_logL_ggm
                            best_result_dict = result_dict
                    
                except Exception as e:
                    print(f"\n{center, scale, max_weight} 조합에서 오류 발생: {e}")
                    traceback.print_exc()

    search_bar.close()
    
    # 결과 저장
    if all_results:
        # results_df = pd.DataFrame(all_results)
        # output_file = Path(filepath).parent / "가중치 측정 결과.xlsx"
        # results_df.to_excel(output_file, index=False)
        # print(f"\n모든 탐색 결과가 '{output_file}'에 저장되었습니다.")

        if best_result_dict:
            print("\n[탐색 결과 요약]")
            if best_result_dict['logL_diff'] > 0:
                print(f"★ 기존 값 대비 개선 성공! (차이: {best_result_dict['logL_diff']:.10f})")
            else:
                print(f"기존 값 대비 개선 실패 (최소 차이: {new_logL_ggm - baseline_logL_ggm})")
            
            print(f"최적 가중치: center={best_result_dict['center']}, scale={best_result_dict['scale']}, max_weight={best_result_dict['max_weight']}")
            print(f"최적 GGM 파라미터: a={best_result_dict['a']:.6f}, b={best_result_dict['b']:.6f}, gamma={best_result_dict['gamma']:.6f}, c={best_result_dict['c']:.6f}")
    else:
        print("\n탐색 결과가 없습니다.")
    
    return
    
def run_refine_excel(year, sex, Dx, Ex, filepath, observed_mu, bounds = None):

    # 1. 엑셀 파일에서 기존 파라미터 및 로그우도 불러오기
    scale_row = get_data_from_file(filepath, year, sex)
    
    # 필수 파라미터들이 존재하는지 확인
    required_keys = ['a', 'b', 'gamma', 'c', 'center', 'scale', 'max_weight', 'logL_ggm']
    if not all(key in scale_row and scale_row[key] is not None for key in required_keys):
        print(f"오류: {year}년 {sex} 데이터에 필수 파라미터가 부족")
        return False
        
    # 기존 파라미터와 로그우도
    existing_params = [scale_row['a'], scale_row['b'], scale_row['gamma'], scale_row['c']]
    existing_logL_ggm = scale_row['logL_ggm']
    weight_params = {
        'center': scale_row['center'],
        'scale': scale_row['scale'],
        'max_weight': scale_row['max_weight']
    }
    try:
        # 가중치 포함 로그우도 함수 생성
        neg_log_likelihood = make_neg_log_likelihood(Dx, Ex, age, weight_func = weight_sigmoid, weight_params = weight_params)
        # minimize 함수에 기존 파라미터를 초기값으로 전달
        bounds = bounds if bounds is not None else [(1e-4, 3e-3), (0.08, 0.14), (0.01, 0.3), (3e-5, 3e-3)] 
        result = minimize(
            fun = neg_log_likelihood,
            x0 = existing_params,
            bounds = bounds,
            method = 'L-BFGS-B'
        )
        
        if not result.success:
            print(f"{year}년 {sex} minimize 최적화에 실패")
            return False
            
        new_params = result.x
        
    except Exception as e:
        print(f"오류: {year}년 {sex} minimize 실행 중 예외 발생 - {e}")
        traceback.print_exc()
        return False

    # 4. 새로운 로그우도 계산 (가중치가 없는 순수 로그우도)
    neg_log_likelihood_pure = make_neg_log_likelihood(Dx, Ex, age, weight_func=None)
    new_logL_ggm = -neg_log_likelihood_pure(new_params)

    # 5. 기존 결과와 비교
    if new_logL_ggm > existing_logL_ggm:
        print(f"개선 성공: 기존 logL({existing_logL_ggm:.4f}) -> 새로운 logL({new_logL_ggm:.4f})")
        
        # 새로운 파라미터로 지표 재계산
        fitted_mu, x_star = calc_ggm(new_params, age)
        metrics = evaluate_fit_metrics(observed_mu, fitted_mu, notice = False)

        # 엑셀에 저장할 딕셔너리 생성
        update_values = {
            "a": new_params[0], "b": new_params[1], "gamma": new_params[2], "c": new_params[3],
            "logL_ggm": new_logL_ggm, "x*": x_star,
            **metrics  # rmse, mae, mape 추가
        }
        
        # 엑셀 파일 업데이트
        save_result_to_excel(update_values, year, sex, filepath)
        
        return True
        
    else:
        print(f"개선 실패, 로그우도 차이: {existing_logL_ggm - new_logL_ggm:.4f}")
        return False


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