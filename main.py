import traceback
import pandas as pd
from tqdm import trange
import func
from pathlib import Path
from plyer import notification
import os
import numpy as np


base_dir = Path(__file__).resolve().parent

# 생명표 읽기
life_table_path = base_dir / "65이상 생명표.xlsx"
df = pd.read_excel(life_table_path, sheet_name = "Sheet1")

output_path_result = base_dir / "측정 결과.xlsx"


year, sex, Dx, Ex, age, observed_mu = func.load_life_table(year = 2001, sex = "남자")

func.draw_LAR_from_file(year, sex)

# func.run_refine_search(year, sex, output_path_result, 
#                             centers = np.arange(87, 96, 1), scales = np.arange(2.0, 10.1, 0.1), max_weights = np.arange(2, 50, 0.1), 
#                             bounds = [(1e-4, 5e-4), (0.09, 0.13), (0.001, 0.3), (3e-4, 1e-3)])

#func.run_refine_excel(year, sex, Dx, Ex, output_path_result, observed_mu, bounds = [(1e-100, 1), (1e-100, 1), (1e-100, 1), (1e-100, 1)])

#--------------------
# TODO evaluate_fit_metrics에 항목 추가?

# TODO 국내 데이터와 러시아, 우크라이나(사망률 높다고 언급됨, 벨 분포 사용하는 대표적인 예시) 비교해보기, 잘 안되면 분산 - 평균 비교랑 잔차 확인
# TODO HMD 자료로 써보기

# TODO LAR 종 형태 나타나는 파라미터 찾아보기(html 파일 사용)

# TODO GM과의 비교를 꼭 해야할까? 다른 논문에서 언급된거 있으면 가져다가 쓰고 나는 그냥 GGM만 돌리는게 낫지않을까?
# 근데 이러니까 MAPE 망함... 연도별로 boundary 나눠줘야 하나?
#--------------------
# center_range = (85, 96, 1), scale_range = (1.0, 10.1, 0.5), max_weight_range = (2, 20, 1)
# center = previous_result['center'], scale = previous_result['scale'], max_weight = previous_result['max_weight']

# try:
#     previous_result = func.get_data_from_file(output_path_result, year, sex)
#     func.find_best_scale(year, sex, Dx, Ex, age, trial = 100, n_runs = 1,
#             center_range = 87, scale_range = 9, max_weight_range = 2, bounds = [(3e-4, 3e-3), (0.08, 0.14), (0.01, 0.3), (3e-5, 3e-3)],
#                     filepath = output_path_result, notice = True, show_graph = True, compare_gm = False,
#                     best_logL_ggm = previous_result['logL_ggm'], best_logL_gm = previous_result['logL_gm'])

# except AttributeError as e:
#     print(f"결과 저장 실패 - 개선된 결과가 없습니다. ({e})")   

# except Exception as e:
#     traceback.print_exc()
#     print(f"알 수 없는 오류 발생: {e}")     

# finally: 
#     pass
#    os.system("shutdown /h")    
#--------------------

#-------------------- for문 사용할 때
#center_range = (85, 96, 1), scale_range = (1.0, 10.1, 0.5), max_weight_range = (6, 12, 1)

# previous_result = func.get_data_from_file(output_path_result, year, sex) 
# temp_best_logL_ggm = previous_result['logL_ggm'] if previous_result['logL_ggm'] is not None else -np.inf
# temp_best_result = None
# temp_best_scale_params = None
# improve_count = 0

# for i in trange(93, 94, desc = "진행중") :
#     for j in [round(x, 1) for x in np.arange(3, 5.1, 0.5)]:
#         for k in range(7, 11, 1):
#             try:
#                 print(f"\ncenter = {i}, scale = {j}, max weight = {k}")
                
#                 best_result, best_logL_ggm, best_scale_params, result_gm = func.find_best_scale(year = year, sex = sex, trial = 100,  n_runs = 1, 
#                                     center_range = i, scale_range = j, max_weight_range = k, show_graph = False,
#                                     Dx = Dx, Ex = Ex, age = age, filepath = output_path_result, notice = False, compare_gm = False,
#                                     best_logL_ggm = previous_result['logL_ggm'], best_logL_gm = previous_result['logL_gm'])
#                 if best_logL_ggm > temp_best_logL_ggm :
#                     temp_best_logL_ggm = best_logL_ggm
#                     temp_best_result = best_result
#                     temp_best_scale_params = best_scale_params
#                     improve_count += 1
#             except AttributeError as e:
#                 print(f"결과 저장 실패 - 개선된 결과가 없습니다. ({e})")   

#             except Exception as e:
#                 print(f"알 수 없는 오류 발생: {e}")     

# if improve_count == 0 :
#     print("모든 경우에서 결과 개선 실패")
# elif improve_count > 0 :
#     print(f"{improve_count}회 개선")



    # print(f"로그우도 개선 수치: {best_logL_ggm - previous_result['logL_ggm']}")
    # print("최적 scale:")
    # print(f"center     = {best_scale_params['center']}")
    # print(f"scale      = {best_scale_params['scale']}")
    # print(f"max_weight = {best_scale_params['max_weight']}")    
    # print(best_result.x)

# os.system("shutdown /h")                 

#--------------------
# notification.notify(title="작업 완료", timeout=5)
