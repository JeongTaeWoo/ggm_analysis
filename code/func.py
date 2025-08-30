from pathlib import Path
import random
from scipy.optimize import differential_evolution, minimize, dual_annealing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import trange
import traceback

base_dir = Path(__file__).resolve().parent

# 생명표 읽기
life_table_path = base_dir / "전연령 생명표.xlsx"

df = pd.read_excel(life_table_path, sheet_name="Sheet1")
def load_life_table(year, sex, df = df) : # 통계청 생명표와 HMD 생명표는 다른 구조로 읽게 됨
    
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