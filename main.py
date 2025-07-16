import pandas as pd
import func
from pathlib import Path
import os


base_dir = Path(__file__).resolve().parent

# 생명표 읽기
life_table_path = base_dir / "65이상 생명표.xlsx"
df = pd.read_excel(life_table_path, sheet_name="Sheet1")


output_path_batch = base_dir / "적합 결과.xlsx"
output_path_weight = base_dir / "가중치 측정 결과.xlsx"


year, sex, Dx, Ex, age, observed_mu = func.load_excel(year = 2018, sex = "남자")


#--------------------
#func.run_batch(years = range(1990, 2021), #전체 범위는 (1970, 2024)
#               sex = '여자', df = df, trial = 200,
#               center = 80, scale = 3, max_weight = 5,
#               output_path = output_path_batch)
#--------------------

# scale_result = func.get_scale_data_from_file(output_path_weight, year, sex, 
#     default_center = 91, default_scale = 5, default_max_weight = 5)
# func.run_test(year = year, sex = sex, df = df, trial = 500, use_weights = True, notice = True,
#             center = scale_result['center'], scale = scale_result['scale'], max_weight = scale_result['max_weight'], result_path = None,
#             opt_func = "differential_evolution")
#--------------------
# func.run_test(year = year, sex = sex, df = df, trial = 500, use_weights = True, notice = True,
#             center = 91, scale = 9, max_weight = 13, result_path = None,
#             opt_func = "differential_evolution")
#--------------------

#--------------------
#center_range = (85, 96, 1), scale_range = (1.0, 10.1, 0.5), max_weight_range = (2, 20, 1), n_runs = 20,
func.fitting_gm(year = year, sex = sex, age = age)
try:
    scale_result = func.get_scale_data_from_file(output_path_weight, year, sex)
    best_result, best_logL, best_scale_params = func.find_best_scale(year = year, sex = sex, trial = 100, 
                        center_range = (85, 96, 1), scale_range = (1.0, 10.1, 0.5), max_weight_range = (2, 20, 1), n_runs = 30,
                        Dx = Dx, Ex = Ex, age = age, 
                        best_logL = scale_result['logL'])
    func.save_scale_result_to_excel(best_result, best_logL, best_scale_params, year, sex, filepath = output_path_weight)
finally: pass
    #os.system("shutdown /h")    
#--------------------



#--------------------
#result = func.result_maker(1.12E-05,	0.120908336,	0.207022451,	0.0272633321)
#func.fitted_plot(result, mu_obs)
#--------------------
