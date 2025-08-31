import pandas as pd
from pathlib import Path

def load_life_table(key, sex):
    """
    생명표 종류(key)와 성별(sex)에 따라 데이터를 불러오는 함수.

    Args:
        filepath (str): Excel 파일의 경로.
        key (str): 불러올 생명표 종류 ('kr', 'hmd_male', 'hmd_female' 등). 현재 'kr'만 구현.
        sex (str): '남자' 또는 '여자'.

    Returns:
        tuple: (years, age, mortality_df, Dx_df, Ex_df)
            years: 연도 리스트
            age: 연령 리스트
            mortality_df: 사망률 데이터프레임 (인덱스: age, 컬럼: year)
            Dx_df: 사망자 수 데이터프레임
            Ex_df: 정지인구 데이터프레임
    """
    if key == 'kr':
        base_dir = Path(__file__).resolve().parent

        # 생명표 읽기
        filepath = base_dir / "전연령 생명표.xlsx"
        df_raw = pd.read_excel(filepath)

        # 연령을 정수형으로 변환하여 '100세 이상' 행을 NaN으로 만듭니다.
        df_raw['age'] = pd.to_numeric(df_raw['age'], errors='coerce')
        # NaN이 된 '100세 이상' 행과 같은 불필요한 행을 제거합니다.
        df_raw = df_raw.dropna(subset=['age'])
        df_raw['age'] = df_raw['age'].astype(int)

        years = [str(col) for col in df_raw.columns if str(col).isdigit()]
        
        # 생존자(lx)와 정지인구(Ex) 데이터 추출
        df_lx = df_raw[df_raw['title'] == f'생존자({sex})'].set_index('age')[years].astype(float)
        df_Ex = df_raw[df_raw['title'] == f'정지인구({sex})'].set_index('age')[years].astype(float)

        # 사망자 수(Dx) 계산: Dx = lx - lx+1
        # 99세의 Dx를 계산하기 위해 100세의 lx가 필요하지만,
        # 100세 이상 그룹은 제외되었으므로, 100세의 사망률은 1로 간주하여 Dx를 계산합니다.
        # 기존에 lx[:-1]을 사용했던 사용자 코드와 유사하게, 마지막 행을 제외하고 계산합니다.
        df_lx_calc = df_lx.iloc[:-1]
        df_lx_plus1_calc = df_lx.iloc[1:]
        
        df_Dx = df_lx_calc - df_lx_plus1_calc.set_index(df_lx_calc.index)

        # 사망률(observed_mu) 계산: observed_mu = Dx / Ex
        df_Ex_calc = df_Ex.iloc[:-1]
        df_mu = df_Dx / df_Ex_calc.set_index(df_Dx.index)
        
        # 유효한 연령 리스트를 반환합니다.
        ages = df_mu.index.values

        return years, ages, df_mu, df_Dx, df_Ex_calc
    
    elif key in ['hmd_male', 'hmd_female']:
        raise NotImplementedError(f"HMD 데이터셋 ('{key}') 처리는 아직 구현되지 않았습니다.")
    
    else:
        raise ValueError(f"지원하지 않는 key 값입니다: {key}")

