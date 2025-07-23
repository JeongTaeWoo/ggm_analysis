import pandas as pd
import func
from pathlib import Path
from plyer import notification
import os


base_dir = Path(__file__).resolve().parent

# 생명표 읽기
life_table_path = base_dir / "65이상 생명표.xlsx"
df = pd.read_excel(life_table_path, sheet_name = "Sheet1")

output_path_result = base_dir / "측정 결과.xlsx"


year, sex, Dx, Ex, age, observed_mu = func.load_life_table(year = 2015, sex = "남자")

# TODO run_test 작동 뭔가 이상하니 이것저것 좀 손봐야 할듯? GPT한테 주니까 개판으로 만들어버림
#--------------------
# center = previous_result['center'], scale = previous_result['scale'], max_weight = previous_result['max_weight']
# previous_result = func.get_data_from_file(output_path_result, year, sex)
# best_result, best_logL, best_scale_params, result_gm = func.run_test(year = year, sex = sex, df = df, trial = 500, use_weights = True, notice = True,
#             center = previous_result['center'], scale = previous_result['scale'], max_weight = previous_result['max_weight'], result_path = None,
#             opt_func = "differential_evolution")
# func.save_result_to_excel(best_result, result_gm, best_logL, best_scale_params, year, sex, filepath = output_path_result)
#--------------------
# func.run_test(year = year, sex = sex, df = df, trial = 500, use_weights = True, notice = True,
#             center = 91, scale = 9, max_weight = 13, result_path = None,
#             opt_func = "differential_evolution", fallback_filepath = output_path_result)
#--------------------
# TODO fit_ggm 함수의 weight_params랑 rmse_params 통일시키기
#--------------------
#center_range = (85, 96, 1), scale_range = (1.0, 10.1, 0.5), max_weight_range = (2, 20, 1), n_runs = 20,
try:
    previous_result = func.get_data_from_file(output_path_result, year, sex) ; print(previous_result)
    best_result, best_logL, best_scale_params, result_gm = func.find_best_scale(year = year, sex = sex, trial = 10, 
                        center_range = (89, 94, 1), scale_range = (1.0, 10.1, 0.5), max_weight_range = (2, 20, 1), n_runs = 4,
                        Dx = Dx, Ex = Ex, age = age, filepath = output_path_result,
                        best_logL_ggm = previous_result['logL_ggm'], best_logL_gm = previous_result['logL_gm'])
    func.save_result_to_excel(best_result, result_gm, best_logL, best_scale_params, year, sex, filepath = output_path_result)

except AttributeError as e:
    print(f"결과 저장 실패 - 개선된 결과가 없습니다. ({e})")   

except Exception as e:
    print(f"알 수 없는 오류 발생: {e}")     

finally: 
    pass
    #os.system("shutdown /h")    
#--------------------

# TODO i j k 정수밖에 못 가지는듯? scale 0.5단위로 하고싶음
#-------------------- for문 사용할 때
#center_range = (85, 96, 1), scale_range = (1.0, 10.1, 0.5), max_weight_range = (2, 20, 1), n_runs = 20,
# result_gm = func.fitting_gm(year = year, sex = sex, age = age, show_graph = False)
# for i in range(89, 93, 1) :
#     for j in range(1, 10, 1):
#         for k in range(2, 20, 1):
#             try:
#                 print(f"center = {i}, scale = {j}, max weight = {k}")
#                 previous_result = func.get_data_from_file(output_path_result, year, sex) 
#                 best_result, best_logL, best_scale_params, result_gm = func.find_best_scale(year = year, sex = sex, trial = 50, 
#                                     center_range = i, scale_range = j, max_weight_range = k, n_runs = 2, show_graph = False,
#                                     Dx = Dx, Ex = Ex, age = age, filepath = output_path_result,
#                                     best_logL_ggm = previous_result['logL_ggm'], best_logL_gm = previous_result['logL_gm'])
#                 func.save_result_to_excel(best_result, result_gm, best_logL, best_scale_params, year, sex, filepath = output_path_result)


#             except AttributeError as e:
#                 print(f"결과 저장 실패 - 개선된 결과가 없습니다. ({e})")   

#             except Exception as e:
#                 print(f"알 수 없는 오류 발생: {e}")     

#             finally: 
#                 pass
# #os.system("shutdown /h")                 
#--------------------

#--------------------
#result = func.result_maker(1.12E-05,	0.120908336,	0.207022451,	0.0272633321)
#func.fitted_plot(result, mu_obs)
#--------------------
notification.notify(title="작업 완료", timeout=5)
