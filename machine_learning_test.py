# 필요한 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

# func.py에서 필요한 함수 임포트
# func.py 파일이 현재 실행 환경에서 접근 가능한 경로에 있어야 합니다.
# 예를 들어, 같은 디렉토리에 func.py가 있다면 이대로 사용할 수 있습니다.
from func import load_life_table
plt.rcParams['font.family'] = 'Malgun Gothic' # 폰트 설정
plt.rcParams['axes.unicode_minus'] = False # 마이너스 부호 깨짐 방지

# 0. 사용자 XLSX 파일 불러오기
# 사용자가 업로드한 파일의 원래 이름인 '전연령 생명표.xlsx'를 사용합니다.
# pandas.read_excel을 사용하여 XLSX 파일을 읽습니다.
base_dir = Path(__file__).resolve().parent
# 생명표 읽기
file_path_excel = base_dir / '전연령 생명표.xlsx'
df_life_table_raw = pd.read_excel(file_path_excel, sheet_name="Sheet1") # 'Sheet1' 시트를 읽도록 가정

# 1. 데이터 전처리 및 FNN 모델을 위한 데이터셋 구성
all_data = []

# 분석할 연도 및 성별 리스트 (XLSX 파일에 있는 모든 연도와 성별을 사용)
# 연도 컬럼을 동적으로 찾기 (title, age를 제외한 컬럼)
year_columns = [col for col in df_life_table_raw.columns if col not in ['age', 'title']]
years_to_process = sorted([int(y) for y in year_columns if str(y).isdigit()]) # 숫자형 연도만 추출

# '사망확률(남자)', '사망확률(여자)' 데이터를 찾아서 처리
# func.py의 load_life_table 함수는 '생존자(남자)'를 기준으로 age를 추출하고,
# '사망확률(성별)'을 기준으로 mortality_rate를 추출합니다.
genders = ['남자', '여자'] # XLSX 파일의 title 컬럼에서 '사망확률(남자)', '사망확률(여자)'를 추출하여 사용

for year in years_to_process:
    for gender in genders:
        # func.py의 load_life_table 함수를 사용하여 데이터 로드
        # func.py의 load_life_table은 '생존자(남자)'로 age를 추출하고,
        # '사망확률(성별)'로 observed_mu를 추출하도록 내부 로직이 되어 있습니다.
        # 여기서는 사망확률 데이터만 필요하므로, 이 함수를 직접 사용하여 df_life_table_raw에서 원하는 데이터를 추출합니다.
        _, _, _, _, age_arr, observed_mu_arr = load_life_table(str(year), gender, df_life_table_raw) # year를 문자열로 전달

        for i in range(len(age_arr)):
            all_data.append({
                'age': age_arr[i],
                'year': year,
                'gender': gender,
                'mortality_rate': observed_mu_arr[i]
            })

df_processed = pd.DataFrame(all_data)
df_processed.dropna(inplace=True) # NaN 값 제거

# 2. 피드포워드 신경망(FNN) 모델을 위한 데이터 준비

# 입력 변수(features)와 목표 변수(target) 설정
features = df_processed[['age', 'year', 'gender']]
target = df_processed['mortality_rate']

# 범주형 변수(gender)에 대한 원-핫 인코딩
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
gender_encoded = encoder.fit_transform(features[['gender']])
gender_df = pd.DataFrame(gender_encoded, columns=encoder.get_feature_names_out(['gender']), index=features.index)

# 숫자형 변수(age, year) 정규화
scaler = StandardScaler()
numerical_features = features[['age', 'year']]
numerical_scaled = scaler.fit_transform(numerical_features)
numerical_df = pd.DataFrame(numerical_scaled, columns=['age', 'year'], index=features.index)

# 모든 변수 결합
X = pd.concat([numerical_df, gender_df], axis=1)
y = target

# 3. 학습 및 예측 성능 평가 (홀드아웃 검증)

# 2018년 이후 데이터를 테스트 데이터로 분할 (홀드아웃 검증 시나리오)
test_year_start = 2018
X_train = X[features['year'] < test_year_start]
X_test = X[features['year'] >= test_year_start]
y_train = y[features['year'] < test_year_start]
y_test = y[features['year'] >= test_year_start]

# FNN 모델 구축
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)), # input_shape 추가
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1) # 회귀를 위한 출력층
])

# 모델 컴파일
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# 모델 학습
print("\n--- FNN 모델 학습 시작 ---")
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, verbose=0)
print("--- FNN 모델 학습 완료 ---\n")

# 모델 성능 평가
y_pred = model.predict(X_test).flatten()
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"모델 평가 결과 ({test_year_start}년~2023년 데이터):")
print(f"Mean Absolute Error (MAE): {mae:.6f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")

# 4. 결과 시각화
# 특정 나이와 성별에 대한 예측 곡선 시각화
plt.figure(figsize=(15, 6))
plt.style.use('seaborn-v0_8-whitegrid')

# 시각화할 나이와 성별 지정
selected_age = 60 # 예시로 60세 선택
selected_gender = '남자'

# 해당 나이, 성별에 대한 실제 데이터 추출
actual_data_plot = df_processed[(df_processed['age'] == selected_age) & (df_processed['gender'] == selected_gender)].sort_values('year')

# 해당 나이, 성별에 대한 예측 데이터 준비
# 전체 연도에 대한 예측을 위해 원본 스케일링/인코딩을 그대로 사용
years_for_prediction = sorted(df_processed['year'].unique())
ages_for_prediction = [selected_age] * len(years_for_prediction)
genders_for_prediction = [selected_gender] * len(years_for_prediction)

prediction_df = pd.DataFrame({
    'age': ages_for_prediction,
    'year': years_for_prediction,
    'gender': genders_for_prediction
})

# 동일한 스케일러와 인코더 사용
prediction_gender_encoded = encoder.transform(prediction_df[['gender']])
prediction_gender_df = pd.DataFrame(prediction_gender_encoded, columns=encoder.get_feature_names_out(['gender']))
prediction_numerical_scaled = scaler.transform(prediction_df[['age', 'year']])
prediction_numerical_df = pd.DataFrame(prediction_numerical_scaled, columns=['age', 'year'])

X_prediction = pd.concat([prediction_numerical_df, prediction_gender_df], axis=1)
predicted_mortality_full = model.predict(X_prediction).flatten()

# 그래프 그리기
plt.plot(actual_data_plot['year'], actual_data_plot['mortality_rate'], 'o-', label='실제 사망확률', color='#1f77b4')
plt.plot(years_for_prediction, predicted_mortality_full, 's--', label='예측 사망확률', color='#ff7f0e')

# 테스트 데이터 기간 표시
plt.axvline(x=test_year_start, color='red', linestyle='--', label=f'학습/테스트 데이터 분할 ({test_year_start}년)')

plt.title(f'{selected_age}세 {selected_gender} 사망확률 예측 곡선 비교', fontsize=16)
plt.xlabel('연도', fontsize=12)
plt.ylabel('사망확률', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True)
plt.show()

