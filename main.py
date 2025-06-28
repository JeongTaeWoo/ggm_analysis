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
df = pd.read_excel(file_path, sheet_name="Sheet1")


df_surv = df[df['title'] == '생존자(남자)']
age_raw = pd.to_numeric(df_surv['age'], errors='coerce')
age = age_raw[:-1].reset_index(drop=True)


df_exp = df[df['title'] == '정지인구(남자)']

year = 1970

lx_raw = pd.to_numeric(df_surv[year], errors='coerce')
Ex_raw = pd.to_numeric(df_exp[year], errors='coerce')

lx = lx_raw[:-1].reset_index(drop=True)
lx_plus1 = lx_raw[1:].reset_index(drop=True)
Ex = Ex_raw[:-1].reset_index(drop=True)
Dx = lx - lx_plus1


# 유효한 값만 필터링
valid = (~Dx.isna()) & (~Ex.isna()) & (Dx >= 0) & (Ex > 0)
age = age[valid].reset_index(drop=True)
Dx = Dx[valid].reset_index(drop=True)
Ex = Ex[valid].reset_index(drop=True)

func.age = age; func.Dx = Dx; func.Ex = Ex

mu_obs = Dx / Ex

init_params = (0.00005,	0.1, 0.1, 0.0001)
#result = func.fit_ggm_mle(age, Dx, Ex, mu_obs, init_params)
#func.fitted_plot(result, mu_obs)
#func.draw_LAR(params = init_params)

#func.gm_mle_fit(age, Dx, Ex, mu_obs)


#func.run_batch(years = range(1990, 2021), #전체 범위는 (1970, 2024)
#               sex = '여자', df = df, trial = 200,
#               center = 80, scale = 3, max_weight = 5,
#               output_path = "C:/Users/tw010/Desktop/project/적합 결과.csv")
               
               
# TODO GM 그래프도 같이 띄우고 RMSE 평가치도 바로 나오게끔 하자

# TODO 아예 center, scale, max_weight도 랜덤으로 돌려버릴까? 
# 가장 적합도 좋았던 결과의 모수도 같이 나오게 하면 되잖아
#func.run_test(year = 2018, sex = '남자', df = df, trial = 100, 
#              center = 90, scale = 3, max_weight = 10)

func.run_test(year = 2001, sex = '남자', df = df, trial = 500, use_weights = False,
              center = 85, scale = 3, max_weight = 10, result_path = None,
              opt_func = "differential_evolution")


#result = func.result_maker(1.12E-05,	0.120908336,	0.207022451,	0.0272633321)
#func.fitted_plot(result, mu_obs)


"""
opt_func 목록

differential_evolution
dual_annealing
"""

