import pandas as pd
import func
from pathlib import Path


base_dir = Path(__file__).resolve().parent

# 생명표 읽기
life_table_path = base_dir / "65이상 생명표.xlsx"
df = pd.read_excel(life_table_path, sheet_name="Sheet1")


output_path_batch = base_dir / "적합 결과.xlsx"
output_path_weight = base_dir / "가중치 측정 결과.xlsx"


year, sex, Dx, Ex, age, observed_mu = func.load_excel(year = 2014, sex = "여자")

#--------------------
#func.run_batch(years = range(1990, 2021), #전체 범위는 (1970, 2024)
#               sex = '여자', df = df, trial = 200,
#               center = 80, scale = 3, max_weight = 5,
#               output_path = output_path_batch)
#--------------------
# TODO 여자 GM 적합 이상함
#--------------------
#func.run_test(year = year, sex = sex, df = df, trial = 1000, use_weights = True, notice = True,
#            center = 87, scale = 8.5, max_weight = 10, result_path = None,
#            opt_func = "differential_evolution")
#--------------------

#--------------------
best_result, best_logL, best_scale_params = func.find_best_scale(year = year, sex = sex, trial = 50, 
                    center_range = (85, 96, 1), scale_range = (1.0, 10.1, 0.5), max_weight_range = (2, 20, 1), n_runs = 50,
                    Dx = Dx, Ex = Ex, age = age, 
                    best_logL = func.get_best_logL_from_file(output_path_weight, year, sex))
func.save_scale_result_to_excel(best_result, best_logL, best_scale_params, year, sex, filepath = output_path_weight)
#--------------------



#--------------------
#result = func.result_maker(1.12E-05,	0.120908336,	0.207022451,	0.0272633321)
#func.fitted_plot(result, mu_obs)
#--------------------

#--------------------
#neg_log_likelihood_pure = func.make_neg_log_likelihood(Dx, Ex, age, weight_func = None)
#params = (0.00002811,0.10332199,0.10251986,0.00081814)
#logL_ggm_pure = -neg_log_likelihood_pure(params)
#print(logL_ggm_pure)
#--------------------
"""
opt_func 목록

differential_evolution
dual_annealing
"""

