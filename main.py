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


year, sex, Dx, Ex, age, observed_mu = func.load_life_table(year = 2011, sex = "남자")

#--------------------
# TODO evaluate_fit_metrics에 항목 추가? 어떤거?

# TODO 논문 읽었던거 내용 간단하게라도 정리해서 모아두기 - 진행중

# TODO 엑셀 업데이트 많이 됐으니까 엑셀 읽어다가 그래프 그려주고 + metrics 보여주는 함수 만들기

# TODO GM과의 비교를 꼭 해야할까? 다른 논문에서 언급된거 있으면 가져다가 쓰고 나는 그냥 GGM만 돌리는게 낫지않을까?
# 근데 이러니까 MAPE 망함

# TODO RMSE filter 기능 제거하기 (어차피 엑셀에서 결과 다 보임)
#--------------------
# center_range = (85, 96, 1), scale_range = (1.0, 10.1, 0.5), max_weight_range = (2, 20, 1)
# center = previous_result['center'], scale = previous_result['scale'], max_weight = previous_result['max_weight']

# try:
#     previous_result = func.get_data_from_file(output_path_result, year, sex)
#     func.find_best_scale(year = year, sex = sex, trial = 500, n_runs = 3,
#                         center_range = 91, scale_range = 3, max_weight_range = 15,
#                         Dx = Dx, Ex = Ex, age = age, filepath = output_path_result, notice = True, compare_gm = False,
#                         best_logL_ggm = previous_result['logL_ggm'], best_logL_gm = previous_result['logL_gm'])

# except AttributeError as e:
#     print(f"결과 저장 실패 - 개선된 결과가 없습니다. ({e})")   

# except Exception as e:
#     traceback.print_exc()
#     print(f"알 수 없는 오류 발생: {e}")     

# finally: 
#     pass
# #    os.system("shutdown /h")    
#--------------------

#-------------------- for문 사용할 때
#center_range = (85, 96, 1), scale_range = (1.0, 10.1, 0.5), max_weight_range = (6, 12, 1)

previous_result = func.get_data_from_file(output_path_result, year, sex) 
temp_best_logL_ggm = previous_result['logL_ggm'] if previous_result['logL_ggm'] is not None else -np.inf
temp_best_result = None
temp_best_scale_params = None
improve_count = 0

for i in trange(93, 94, desc = "진행중") :
    for j in [round(x, 1) for x in np.arange(3, 5.1, 0.5)]:
        for k in range(7, 11, 1):
            try:
                print(f"\ncenter = {i}, scale = {j}, max weight = {k}")
                
                best_result, best_logL_ggm, best_scale_params, result_gm = func.find_best_scale(year = year, sex = sex, trial = 100,  n_runs = 1, 
                                    center_range = i, scale_range = j, max_weight_range = k, show_graph = False,
                                    Dx = Dx, Ex = Ex, age = age, filepath = output_path_result, notice = False, compare_gm = False,
                                    best_logL_ggm = previous_result['logL_ggm'], best_logL_gm = previous_result['logL_gm'])
                if best_logL_ggm > temp_best_logL_ggm :
                    temp_best_logL_ggm = best_logL_ggm
                    temp_best_result = best_result
                    temp_best_scale_params = best_scale_params
                    improve_count += 1
            except AttributeError as e:
                print(f"결과 저장 실패 - 개선된 결과가 없습니다. ({e})")   

            except Exception as e:
                print(f"알 수 없는 오류 발생: {e}")     

if improve_count == 0 :
    print("모든 경우에서 결과 개선 실패")
elif improve_count > 0 :
    print(f"{improve_count}회 개선")



    # print(f"로그우도 개선 수치: {best_logL_ggm - previous_result['logL_ggm']}")
    # print("최적 scale:")
    # print(f"center     = {best_scale_params['center']}")
    # print(f"scale      = {best_scale_params['scale']}")
    # print(f"max_weight = {best_scale_params['max_weight']}")    
    # print(best_result.x)

# os.system("shutdown /h")                 

#--------------------
# notification.notify(title="작업 완료", timeout=5)
