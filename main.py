import traceback
import pandas as pd
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


year, sex, Dx, Ex, age, observed_mu = func.load_life_table(year = 2019, sex = "여자")

#--------------------
# TODO evaluate_fit_metrics에 항목 추가? 어떤거?

# TODO 논문 읽었던거 내용 간단하게라도 정리해서 모아두기 - 진행중

# TODO 엑셀 업데이트 많이 됐으니까 엑셀 읽어다가 그래프 그려주고 + metrics 보여주는 함수 만들기

# TODO 이것저것 개선은 했는데 여전히 여자 데이터 결과가 안좋음. scale parameter 다시한번 찾아보기
#--------------------
# center_range = (85, 96, 1), scale_range = (1.0, 10.1, 0.5), max_weight_range = (2, 20, 1)
# center = previous_result['center'], scale = previous_result['scale'], max_weight = previous_result['max_weight']

try:
    previous_result = func.get_data_from_file(output_path_result, year, sex)
    func.find_best_scale(year = year, sex = sex, trial = 1000, n_runs = 1,
                        center_range = 87, scale_range = 4, max_weight_range = 9,
                        Dx = Dx, Ex = Ex, age = age, filepath = output_path_result, notice = True,
                        best_logL_ggm = previous_result['logL_ggm'], best_logL_gm = previous_result['logL_gm'])

except AttributeError as e:
    print(f"결과 저장 실패 - 개선된 결과가 없습니다. ({e})")   

except Exception as e:
    traceback.print_exc()
    print(f"알 수 없는 오류 발생: {e}")     

finally: 
    pass
#    os.system("shutdown /h")    
#--------------------

#-------------------- for문 사용할 때
#center_range = (85, 96, 1), scale_range = (1.0, 10.1, 0.5), max_weight_range = (6, 12, 1), n_runs = 20,

# for i in range(87, 89) :
#     for j in [round(x, 1) for x in np.arange(2, 5.1, 0.5)]:
#         for k in range(5, 10, 1):
#             try:
#                 print(f"center = {i}, scale = {j}, max weight = {k}")
#                 previous_result = func.get_data_from_file(output_path_result, year, sex) 
#                 func.find_best_scale(year = year, sex = sex, trial = 200,  n_runs = 1, 
#                                     center_range = i, scale_range = j, max_weight_range = k, show_graph = False,
#                                     Dx = Dx, Ex = Ex, age = age, filepath = output_path_result, notice = False,
#                                     best_logL_ggm = previous_result['logL_ggm'], best_logL_gm = previous_result['logL_gm'])
#             except AttributeError as e:
#                 print(f"결과 저장 실패 - 개선된 결과가 없습니다. ({e})")   

#             except Exception as e:
#                 print(f"알 수 없는 오류 발생: {e}")     

#             finally: 
#                 pass
#os.system("shutdown /h")                 

#--------------------
# notification.notify(title="작업 완료", timeout=5)
