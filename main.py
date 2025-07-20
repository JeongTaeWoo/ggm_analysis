import pandas as pd
import func
from pathlib import Path
from plyer import notification


base_dir = Path(__file__).resolve().parent


# 생명표 읽기
life_table_path = base_dir / "65이상 생명표.xlsx"
df = pd.read_excel(life_table_path, sheet_name="Sheet1")


output_path_batch = base_dir / "적합 결과.xlsx"
output_path_weight = base_dir / "가중치 측정 결과.xlsx"


year, sex, Dx, Ex, age, observed_mu = func.load_life_table(year = 2014, sex = "남자")


#--------------------
#func.run_batch(years = range(1990, 2021), #전체 범위는 (1970, 2024)
#               sex = sex, df = df, trial = 200,
#               center = 80, scale = 3, max_weight = 5,
#               output_path = output_path_batch)
#--------------------

#--------------------
# scale_result = func.get_scale_data_from_file(output_path_weight, year, sex, 
#     default_center = 91, default_scale = 5, default_max_weight = 5)
# func.run_test(year = year, sex = sex, df = df, trial = 500, use_weights = True, notice = True,
#             center = scale_result['center'], scale = scale_result['scale'], max_weight = scale_result['max_weight'], result_path = None,
#             opt_func = "differential_evolution")
#--------------------
# func.run_test(year = year, sex = sex, df = df, trial = 500, use_weights = True, notice = True,
#             center = 91, scale = 9, max_weight = 13, result_path = None,
#             opt_func = "differential_evolution", fallback_filepath = output_path_weight)
#--------------------

#--------------------
#center_range = (85, 96, 1), scale_range = (1.0, 10.1, 0.5), max_weight_range = (2, 20, 1), n_runs = 20,
try:
    scale_result = func.get_scale_data_from_file(output_path_weight, year, sex) ; print(scale_result)
    best_result, best_logL, best_scale_params, result_gm = func.find_best_scale(year = year, sex = sex, trial = 100, 
                        center_range = (88, 94, 1), scale_range = (1.0, 10.1, 0.5), max_weight_range = (2, 20, 1), n_runs = 30,
                        Dx = Dx, Ex = Ex, age = age, filepath = output_path_weight,
                        best_logL_ggm = scale_result['logL_ggm'], best_logL_gm = scale_result['logL_gm'])
    func.save_scale_result_to_excel(best_result, result_gm, best_logL, best_scale_params, year, sex, filepath = output_path_weight)

except AttributeError as e:
    print(f"결과 저장 실패 - 개선된 결과가 없습니다. ({e})")   

except Exception as e:
    print(f"알 수 없는 오류 발생: {e}")     

finally: 
    pass
    #os.system("shutdown /h")    
#--------------------

#-------------------- for문 사용할 때
#center_range = (85, 96, 1), scale_range = (1.0, 10.1, 0.5), max_weight_range = (2, 20, 1), n_runs = 20,
# result_gm = func.fitting_gm(year = year, sex = sex, age = age)
# for k in range(2, 20, 1):
#     try:
#         scale_result = func.get_scale_data_from_file(output_path_weight, year, sex)
#         best_result, best_logL, best_scale_params = func.find_best_scale(year = year, sex = sex, trial = 100, 
#                             center_range = 91, scale_range = 3.5, max_weight_range = k, n_runs = 5,
#                             Dx = Dx, Ex = Ex, age = age, 
#                             best_logL = scale_result['logL_ggm'])
#         func.save_scale_result_to_excel(best_result, result_gm, best_logL, best_scale_params, year, sex, filepath = output_path_weight)

#     except AttributeError as e:
#         print(f"결과 저장 실패 - 개선된 결과가 없습니다. ({e})")   

#     except Exception as e:
#         print(f"알 수 없는 오류 발생: {e}")     

#     finally: 
#         pass
#         #os.system("shutdown /h") 
#--------------------

#--------------------
#result = func.result_maker(1.12E-05,	0.120908336,	0.207022451,	0.0272633321)
#func.fitted_plot(result, mu_obs)
#--------------------
notification.notify(title="작업 완료", timeout=5)