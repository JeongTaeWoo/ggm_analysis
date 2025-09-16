import pandas as pd
from pathlib import Path

def load_life_table(key: str, sex: str):
    """
    엑셀 파일에서 생명표 데이터를 읽어 사망인구(Dx), 정지인구(Ex),
    관측 사망률(mu)을 계산하여 반환하는 함수입니다.

    Args:
        key (str): 'kr'인 경우에만 파일 경로를 설정합니다.
        sex (str): '남자' 또는 '여자' 중 하나로, 계산할 성별을 지정합니다.

    Returns:
        tuple: (years, ages, mu, Dx, Ex) 튜플을 반환합니다.
        - years (list): 데이터에 포함된 연도 리스트
        - ages (pd.Index): 데이터에 포함된 연령 인덱스
        - mu (pd.DataFrame): 관측 사망률 데이터프레임
        - Dx (pd.DataFrame): 사망인구 데이터프레임
        - Ex (pd.DataFrame): 정지인구 데이터프레임
    """
    # 2. 요구 조건에 따른 시작 부분
    if key == 'kr':
        base_dir = Path(__file__).resolve().parent

        # 생명표 파일 읽기
        # 참고: 사용자가 업로드한 파일은 '전연령 생명표.xlsx - Sheet1.csv' 입니다.
        # 따라서 pd.read_excel 대신 pd.read_csv를 사용합니다.
        # 만약 엑셀 파일(.xlsx)을 사용하시려면 아래 주석 처리된 라인을 사용하고,
        # 파일명을 '전연령 생명표.xlsx'로 변경해주세요.
        # filepath = base_dir / "전연령 생명표.xlsx"
        # df_raw = pd.read_excel(filepath)
        
        filepath = base_dir / "전연령 생명표.xlsx"
        df_raw = pd.read_excel(filepath)

        # 연도 컬럼 추출 (age, title 제외)
        years = [col for col in df_raw.columns if col not in ['age', 'title']]

        # 3. 엑셀에서 인자로 받은 성별에 대한 사망인구 Dx, 정지인구 Ex 계산
        df_Dx = df_raw[df_raw['title'] == f'사망자({sex})'].set_index('age')[years].astype(float)
        df_Ex = df_raw[df_raw['title'] == f'정지인구({sex})'].set_index('age')[years].astype(float)

        # 4. Dx/Ex를 통해 observed mu 계산
        df_mu = df_Dx.divide(df_Ex)

        # 인덱스(연령) 추출
        ages = df_Dx.index

        # 5. years, ages, mu, Dx, Ex 반환
        return years, ages, df_mu, df_Dx, df_Ex
    
    else:
        # 'kr' 이외의 key에 대한 처리 로직을 여기에 추가할 수 있습니다.
        raise ValueError("현재 'kr' 키만 지원됩니다.")