import pandas as pd
import func
import os


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

output_path = "C:/Users/pc/Desktop/project/적합 결과.csv"


Dx, Ex, age, observed_mu = func.load_excel(year = 2021, sex = "남자")


#func.run_batch(years = range(1990, 2021), #전체 범위는 (1970, 2024)
#               sex = '여자', df = df, trial = 200,
#               center = 80, scale = 3, max_weight = 5,
#               output_path = "C:/Users/tw010/Desktop/project/적합 결과.csv")



#func.run_test(year = 2001, sex = '남자', df = df, trial = 1500, use_weights = True, notice = True,
#            center = 89, scale = 1, max_weight = 19, result_path = None,
#            opt_func = "differential_evolution")

# TODO load_excel의 year랑 find_best_scale의 year이 같은지 판별할수 있나??
func.find_best_scale(year = 2021, sex = "남자", trial = 200, 
                    center_range = (85, 96, 1), scale_range = (1.0, 10.0, 0.5), max_weight_range = (2, 20, 1), n_runs = 100,
                    Dx = Dx, Ex = Ex, age = age)
#2001년 남자는 89, 1, 19

#result = func.result_maker(1.12E-05,	0.120908336,	0.207022451,	0.0272633321)
#func.fitted_plot(result, mu_obs)

#neg_log_likelihood_pure = func.make_neg_log_likelihood(Dx, Ex, age, weight_func = None)
#params = (0.00002811,0.10332199,0.10251986,0.00081814)
#logL_ggm_pure = -neg_log_likelihood_pure(params)
#print(logL_ggm_pure)

"""
opt_func 목록

differential_evolution
dual_annealing
"""

