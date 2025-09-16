# 필요한 패키지 불러오기
# install.packages(c("readr", "dplyr", "tidyr", "fda", "forecast", "ggplot2", "readxl", "tidyverse"))
library(readr)
library(dplyr)
library(tidyr)
library(fda)
library(forecast)
library(ggplot2)
library(readxl)
library(tidyverse)
library(stats)

# --- 1단계: 데이터 불러오기 및 전처리 ---
# 엑셀 파일 경로 설정. 사용자 환경에 맞게 수정하세요.
filepath <- "C:/Users/tw010/Desktop/ggm_analysis/code/전연령 생명표.xlsx"
life_table_raw <- read_excel(filepath)

# 데이터 정제 및 성별 분리
life_table_long <- life_table_raw %>%
  pivot_longer(
    cols = as.character(1970:2023),
    names_to = "year",
    values_to = "value"
  ) %>%
  mutate(year = as.numeric(year))

# '사망확률(남자)'와 '사망확률(여자)' 데이터만 필터링하고 age와 year를 기준으로 데이터를 정렬
mortality_male_df <- life_table_long %>%
  filter(grepl("사망확률\\(남자\\)", title)) %>%
  select(age, year, value) %>%
  arrange(age, year)

mortality_female_df <- life_table_long %>%
  filter(grepl("사망확률\\(여자\\)", title)) %>%
  select(age, year, value) %>%
  arrange(age, year)

# 유효한 연령 범위(0-100세) 추출
ages <- 0:100

# 논문과 동일하게 fitting period를 1970-2010년으로 설정
fitting_years <- 1970:2010
mortality_male_fitting <- mortality_male_df %>%
  filter(year %in% fitting_years, age %in% ages) %>%
  pivot_wider(names_from = year, values_from = value) %>%
  column_to_rownames("age")

mortality_female_fitting <- mortality_female_df %>%
  filter(year %in% fitting_years, age %in% ages) %>%
  pivot_wider(names_from = year, values_from = value) %>%
  column_to_rownames("age")

# 예측 기간 설정: 2011년 부터 2023년까지
forecast_start_year <- 2011
forecast_end_year <- 2023
forecast_years <- forecast_start_year:forecast_end_year
n_forecast_years <- length(forecast_years)

# 실제 데이터 (예측값과 비교하기 위함)
mortality_male_actual <- mortality_male_df %>%
  filter(year %in% forecast_years, age %in% ages) %>%
  pivot_wider(names_from = year, values_from = value) %>%
  column_to_rownames("age")

mortality_female_actual <- mortality_female_df %>%
  filter(year %in% forecast_years, age %in% ages) %>%
  pivot_wider(names_from = year, values_from = value) %>%
  column_to_rownames("age")

cat("데이터 로딩 완료.\n")
cat(sprintf("남성 사망률 적합 데이터 형태: %s\n", paste(dim(mortality_male_fitting), collapse=" x ")))
cat(sprintf("여성 사망률 적합 데이터 형태: %s\n", paste(dim(mortality_female_fitting), collapse=" x ")))


# --- 2단계: 데이터 평활화(Smoothing) 및 로그 변환 ---
# 사망률 데이터 로그 변환
log_mortality_male <- log(mortality_male_fitting + 1e-10) # 0 값 방지를 위해 작은 값 더하기
log_mortality_female <- log(mortality_female_fitting + 1e-10)

# 기저 함수 개수를 데이터 포인트 수보다 적게 설정
n_basis <- length(ages) - 1
basis <- create.bspline.basis(rangeval = range(ages), nbasis = n_basis, norder = 4)

smoothed_data_male <- smooth.basis(ages, as.matrix(log_mortality_male), basis)
smoothed_data_female <- smooth.basis(ages, as.matrix(log_mortality_female), basis)

cat("\n데이터 평활화 완료.\n")


# --- 3단계: 함수적 주성분분석(FPCA) 및 4단계: 주성분 점수 시계열 예측 ---
# FDM 모델 구현 함수 (수동 예측)
run_fdm_forecast_manual <- function(smoothed_data, n_components, gender) {
  
  # FPCA 적용
  fpca_result <- pca.fd(smoothed_data, nharm = n_components)
  scores <- fpca_result$scores

  # 주성분 점수 예측 결과를 저장할 매트릭스 초기화
  forecast_scores <- matrix(0, nrow = n_forecast_years, ncol = n_components)

  # 각 주성분 점수에 대해 개별적으로 ARIMA 모델 적합 및 예측
  for (i in 1:n_components) {
    # 주성분 점수 열을 명시적으로 벡터로 변환하여 전달
    arima_model <- auto.arima(as.vector(scores[, i]))
    forecast_scores[, i] <- forecast(arima_model, h = n_forecast_years)$mean
  }

  # 미래 사망률 재구성
  mean_func <- eval.fd(ages, fpca_result$meanfd)
  components_matrix <- eval.fd(ages, fpca_result$harmonics)
  reconstructed_log_mortality <- mean_func + components_matrix %*% t(forecast_scores)
  
  # 원래의 사망률 값으로 되돌리기 위해 exp 변환
  forecast_mortality <- exp(reconstructed_log_mortality)
  
  cat(sprintf("\n--- %s FDM (%d 주성분) 예측 완료 ---\n", gender, n_components))
  return(list(forecast_mortality = forecast_mortality, reconstructed_log_mortality = reconstructed_log_mortality))
}

# 주성분 수에 따른 FDM 모델 실행
n_components_list <- c(1, 2, 3)

for (n_components in n_components_list) {
  
  # 남성 사망률 예측
  result_male <- run_fdm_forecast_manual(smoothed_data_male$fd, n_components, "남성")
  
  # 여성 사망률 예측
  result_female <- run_fdm_forecast_manual(smoothed_data_female$fd, n_components, "여성")
  
  # --- 5단계: 예측 정확도 측정 (MAFE) ---
  mafe_male <- mean(abs(result_male$forecast_mortality - as.matrix(mortality_male_actual)))
  mafe_female <- mean(abs(result_female$forecast_mortality - as.matrix(mortality_female_actual)))
  
  cat(sprintf("\n--- 예측 오차 분석 (2011-2023) ---\n"))
  cat(sprintf("남성 FDM (주성분 수=%d) MAFE: %.6f\n", n_components, mafe_male))
  cat(sprintf("여성 FDM (주성분 수=%d) MAFE: %.6f\n", n_components, mafe_female))
  cat("\n")
}


# --- 6단계: 결과 시각화 ---
# 주성분 수 2개를 사용한 예측 결과를 시각화 예시로 사용
result_male_final <- run_fdm_forecast_manual(smoothed_data_male$fd, 2, "남성")
result_female_final <- run_fdm_forecast_manual(smoothed_data_female$fd, 2, "여성")

# 남성 로그 사망률 시각화
plot_data_male <- data.frame(
  age = rep(ages, length(fitting_years)),
  year = as.factor(rep(fitting_years, each = length(ages))),
  log_mortality = c(as.matrix(log_mortality_male))
)
plot_data_male_forecast <- data.frame(
  age = rep(ages, n_forecast_years),
  year = as.factor(rep(forecast_years, each = length(ages))),
  log_mortality_forecast = c(as.matrix(result_male_final$reconstructed_log_mortality))
)
plot_data_male_actual <- data.frame(
  age = rep(ages, n_forecast_years),
  year = as.factor(rep(forecast_years, each = length(ages))),
  log_mortality_actual = c(as.matrix(log(mortality_male_actual)))
)

ggplot() +
  geom_line(data = plot_data_male, aes(x = age, y = log_mortality, group = year), color = "gray") +
  geom_line(data = plot_data_male_actual, aes(x = age, y = log_mortality_actual, group = year), color = "red", linetype = "solid", linewidth = 1.2) +
  geom_line(data = plot_data_male_forecast, aes(x = age, y = log_mortality_forecast, group = year), color = "blue", linetype = "dashed", linewidth = 1.2) +
  labs(title = "남성 로그 사망률 예측 (FDM)", x = "연령", y = "로그 사망률") +
  theme_minimal() +
  annotate("text", x = 10, y = -12.5, label = "Observed (1970-2010)", color = "gray") +
  annotate("text", x = 10, y = -12, label = "Actual (2011-2023)", color = "red") +
  annotate("text", x = 10, y = -11.5, label = "Forecast (2011-2023)", color = "blue")

# 여성 로그 사망률 시각화
plot_data_female <- data.frame(
  age = rep(ages, length(fitting_years)),
  year = as.factor(rep(fitting_years, each = length(ages))),
  log_mortality = c(as.matrix(log_mortality_female))
)
plot_data_female_forecast <- data.frame(
  age = rep(ages, n_forecast_years),
  year = as.factor(rep(forecast_years, each = length(ages))),
  log_mortality_forecast = c(as.matrix(result_female_final$reconstructed_log_mortality))
)
plot_data_female_actual <- data.frame(
  age = rep(ages, n_forecast_years),
  year = as.factor(rep(forecast_years, each = length(ages))),
  log_mortality_actual = c(as.matrix(log(mortality_female_actual)))
)

ggplot() +
  geom_line(data = plot_data_female, aes(x = age, y = log_mortality, group = year), color = "gray") +
  geom_line(data = plot_data_female_actual, aes(x = age, y = log_mortality_actual, group = year), color = "red", linetype = "solid", linewidth = 1.2) +
  geom_line(data = plot_data_female_forecast, aes(x = age, y = log_mortality_forecast, group = year), color = "blue", linetype = "dashed", linewidth = 1.2) +
  labs(title = "여성 로그 사망률 예측 (FDM)", x = "연령", y = "로그 사망률") +
  theme_minimal() +
  annotate("text", x = 10, y = -12.5, label = "Observed (1970-2010)", color = "gray") +
  annotate("text", x = 10, y = -12, label = "Actual (2011-2023)", color = "red") +
  annotate("text", x = 10, y = -11.5, label = "Forecast (2011-2023)", color = "blue")