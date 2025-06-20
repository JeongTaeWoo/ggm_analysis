library(readr)      # read_csv
library(dplyr)      # 데이터 조작
library(tidyr)      # 필요 시
library(broom)      # lm 결과 정리
library(ggplot2)    # 시각화
library(Metrics)    # rmse, mae, mape 함수
library(readxl)
library(gridExtra)

# 1) 데이터 불러오기
batch <- read_csv("적합 결과_여자_텀페이퍼.csv",
                  locale = locale(encoding = "utf-8"))
life <- read_excel("65이상 생명표.xlsx", sheet = "Sheet1")

# 2) 관측 사망강도 계산 함수
compute_mu_obs <- function(df_life, surv_title, exp_title, year) {
  surv <- df_life %>% 
    filter(title == surv_title) %>% 
    select(age, all_of(as.character(year)))
  exp  <- df_life %>% 
    filter(title == exp_title) %>% 
    select(age, all_of(as.character(year)))
  
  df <- left_join(surv, exp, by = "age", suffix = c("_lx", "_Ex")) %>%
    setNames(c("age", "lx", "Ex")) %>%
    arrange(age) %>%
    mutate(
      Dx = lx - lead(lx),
      mu_obs = Dx / Ex
    ) %>%
    filter(!is.na(mu_obs), Ex > 0, Dx >= 0) %>%
    select(age, mu_obs)
  return(df)
}

# 3) 남자 데이터 long-format 변환
male_results <- batch %>%
  filter(sex == "여자", year >= 1990, year <= 2020) %>%  # ← 이 부분 추가
  pivot_longer(
    cols = matches("^fitted_ggm_\\d+$"),
    names_to = "age",
    names_pattern = "fitted_ggm_(\\d+)",
    values_to = "mu_ggm"
  ) %>%
  mutate(age = as.integer(age)) %>%
  select(year, age, mu_ggm)

# 4) 모든 연도별 관측 사망강도 계산
years <- unique(male_results$year)
mu_obs_list <- lapply(years, function(y) {
  df_obs <- compute_mu_obs(life, "생존자(여자)", "정지인구(여자)", y)
  df_obs <- df_obs %>%
    mutate(
      year = y,
      age  = as.integer(age)        # ★ age를 정수형으로 변환
    ) %>%
    select(year, age, mu_obs)
  return(df_obs)
})

mu_obs_all <- bind_rows(mu_obs_list)
male_with_obs <- male_results %>%
  inner_join(mu_obs_all, by = c("year", "age")) %>%
  filter(!is.na(mu_obs), !is.na(mu_ggm), year >= 1990, year <= 2020)


# 5) GM 로그선형회귀
# mu = a * exp(bx) => log(mu) = log(a) + bx
res_list <- list()
years <- sort(unique(male_with_obs$year))




# ------------------------------------------
for (yr in years) {
  # 연도 추출
  df_year <- male_with_obs %>% filter(year == yr)
  
  # 데이터 개수 충분한지 확인
  if (nrow(df_year) < 5) {
    message(sprintf("Year %d: 관측치 개수 %d개로 부족하여 건너뜁니다.", 
                    yr, nrow(df_year)))
    next
  }
  fit_lm <- try(lm(log(mu_obs) ~ age, data = df_year), silent = True)
  if (inherits(fit_lm, "try-error")) {
    message(sprintf("Year %d: lm 회귀 수행 중 오류 발생, 건너뜁니다.", yr))
    next
  }
  
  # 계수 추출
  tidy_coef <- broom::tidy(fit_lm)
  intercept_est <- tidy_coef %>% filter(term == "(Intercept)") %>% pull(estimate)
  slope_est     <- tidy_coef %>% filter(term == "age")        %>% pull(estimate)
  
  # 파라미터 계산
  a_est <- exp(intercept_est)
  b_est <- slope_est
  
  # GM 모형 계산
  df_year <- df_year %>%
    mutate(mu_gm = a_est * exp(b_est * age),
           a_est = a_est, b_est = b_est)
  
  # 평가 지표 계산
  # RMSE, MAE: GGM 예측(mu_ggm)과 비교
  rmse_gm  <- Metrics::rmse(df_year$mu_obs, df_year$mu_gm)
  rmse_ggm <- Metrics::rmse(df_year$mu_obs, df_year$mu_ggm)
  mae_gm   <- Metrics::mae(df_year$mu_obs, df_year$mu_gm)
  mae_ggm  <- Metrics::mae(df_year$mu_obs, df_year$mu_ggm)
  # MAPE: Metrics::mape은 기본적으로 (abs(obs - pred)/obs) 평균을 반환 (0~1)
  # 여기서는 % 단위로 보기 위해 *100
  mape_gm  <- Metrics::mape(df_year$mu_obs, df_year$mu_gm)  * 100
  mape_ggm <- Metrics::mape(df_year$mu_obs, df_year$mu_ggm) * 100
  
  # 결과 리스트에 저장
  res_list[[as.character(yr)]] <- tibble::tibble(
    year     = yr,
    a_gm     = a_est,
    b_gm     = b_est,
    RMSE_gm  = rmse_gm,
    RMSE_ggm = rmse_ggm,
    MAE_gm   = mae_gm,
    MAE_ggm  = mae_ggm,
    MAPE_gm  = mape_gm,
    MAPE_ggm = mape_ggm
  )
}

df_results <- bind_rows(res_list) %>%
  filter(year >= 1990, year <= 2020)

df_results <- df_results %>%
  mutate(
    RMSE_grade_gm  = sapply(RMSE_gm, rate_rmse),
    RMSE_grade_ggm = sapply(RMSE_ggm, rate_rmse),
    MAE_grade_gm   = sapply(MAE_gm, rate_mae),
    MAE_grade_ggm  = sapply(MAE_ggm, rate_mae),
    MAPE_grade_gm  = sapply(MAPE_gm, rate_mape),
    MAPE_grade_ggm = sapply(MAPE_ggm, rate_mape)
  )

# TODO GM, GGM 모두 병합시키기
# 5) 적합 결과와 병합 
male_with_obs <- male_results %>%
  inner_join(mu_obs_all, by = c("year", "age")) %>%
  filter(!is.na(mu_obs), !is.na(mu_ggm))

# 6) 적합도 지표 계산
quality_male <- male_with_obs %>%
  filter(year >= 1990, year <= 2020) %>%
  group_by(year) %>%
  summarise(
    RMSE = rmse(mu_obs, mu_ggm),
    MAE  = mae(mu_obs, mu_ggm),
    MAPE = mean(abs((mu_obs - mu_ggm)/mu_obs)) * 100,
    .groups = "drop"
  )

# 각 지표에 대해 평가 등급 부여 함수
rate_rmse <- function(x) {
  if (x <= 0.005) "매우 우수"
  else if (x <= 0.01) "양호"
  else "개선 필요"
}

rate_mae <- function(x) {
  if (x <= 0.003) "매우 우수"
  else if (x <= 0.007) "양호"
  else "개선 필요"
}

rate_mape <- function(x) {
  if (x <= 10) "매우 우수"
  else if (x <= 20) "양호"
  else if (x <= 50) "보통"
  else "개선 필요"
}

# 평가 결과에 등급 해석 추가
quality_male_with_grade <- quality_male %>%
  mutate(
    RMSE_해석 = sapply(RMSE, rate_rmse),
    MAE_해석  = sapply(MAE, rate_mae),
    MAPE_해석 = sapply(MAPE, rate_mape)
  ) %>%
  select(year, RMSE_해석, MAE_해석, MAPE_해석)

# 결과 확인
print(quality_male_with_grade, n=60)
# ------------------------------------------




# ------------------------------------------
# 1년짜리 그래프 비교
sel_year <- 2008
df_y <- male_with_obs %>% filter(year == sel_year)
# df_results에서 a_est, b_est 조회
param <- df_results %>% filter(year == sel_year)
if (nrow(df_y) > 0 && nrow(param) == 1) {
  a_e <- param$a_gm; b_e <- param$b_gm
  df_y <- df_y %>% mutate(mu_gm = a_e * exp(b_e * age))
  p <- ggplot(df_y, aes(x = age)) +
    geom_point(aes(y = mu_obs), size=1.5) +
    geom_line(aes(y = mu_gm), color="blue") +
    geom_line(aes(y = mu_ggm), color="red", linetype="dashed") +
    labs(title = paste0("Year ", sel_year, ": 관측 vs Gompertz vs GGM"),
         caption = sprintf("a=%.2e, b=%.3f", a_e, b_e),
         x="Age", y="mu")
  print(p)
}

# 여러 해 그래프 동시에 그리기
plot_gm_ggm_compare <- function(df_results, df_obs, 
                                save_dir = "plots", 
                                sel_year = c(1990,2000,2010,2020)) {
  if (!dir.exists(save_dir)) dir.create(save_dir)
  
  # 1. 연도 목록 지정
  available_years <- sort(unique(df_results$year))
  
  # sel_year 처리: NULL이면 전체, 숫자 벡터면 필터
  if (is.null(sel_year)) {
    years_to_plot <- available_years
  } else if (length(sel_year) == 2 && is.numeric(sel_year)) {
    years_to_plot <- available_years[available_years >= sel_year[1] & available_years <= sel_year[2]]
  } else {
    years_to_plot <- intersect(available_years, sel_year)
  }
  
  # 2. 4개씩 묶어서 그리기
  year_chunks <- split(years_to_plot, ceiling(seq_along(years_to_plot)/4))
  
  for (chunk in year_chunks) {
    plots <- list()
    
    for (yr in chunk) {
      df_y <- df_obs %>% filter(year == yr)
      
      param <- df_results %>% filter(year == yr)
      if (nrow(df_y) == 0 || nrow(param) == 0) next
      
      a_e <- param$a_gm
      b_e <- param$b_gm
      
      df_y <- df_y %>%
        mutate(mu_gm = a_e * exp(b_e * age))
      
      p <- ggplot(df_y, aes(x = age)) +
        geom_point(aes(y = mu_obs), size = 1.5, color = "black", alpha = 0.7) +
        geom_line(aes(y = mu_gm), color = "blue", linewidth = 1) +
        geom_line(aes(y = mu_ggm), color = "red", linewidth = 1, linetype = "dashed") +
        labs(
          title = paste0(yr, " Female"),
          x = "Age",
          y = expression(mu),
        ) +
        theme_minimal(base_size = 12)
      
      plots[[length(plots) + 1]] <- p
    }
    
    # 3. 저장
    if (length(plots) > 0) {
      fname <- paste0(save_dir, "/gm_ggm_comparison_", chunk[1], "_to_", chunk[length(chunk)], ".png")
      png(filename = fname, width = 1600, height = 1200, res = 150)
      gridExtra::grid.arrange(grobs = plots, ncol = 2)
      dev.off()
      message("Saved: ", fname)
    }
  }
}
plot_gm_ggm_compare(df_results, male_with_obs)
# ------------------------------------------




# ------------------------------------------

# 90세 이상 데이터에 대한 수치비교
# 1. GGM 결과와 관측값 결합
ggm_data <- male_with_obs %>%
  mutate(
    model = "GGM",
    mu_pred = mu_ggm
  ) %>%
  select(year, age, mu_obs, mu_pred, model)

# 2. GM 결과와 관측값 결합 (만약 'gm_results'라는 df에 GM 결과가 있다면)
gm_results <- male_with_obs %>%
  left_join(df_results %>% select(year, a_gm, b_gm), by = "year") %>%
  mutate(
    mu_pred = a_gm * exp(b_gm * age)
  ) %>%
  select(year, age, mu_pred)
gm_data <- gm_results %>%
  inner_join(mu_obs_all, by = c("year", "age")) %>%
  mutate(
    model = "GM"
  ) %>%
  select(year, age, mu_obs, mu_pred, model)

# 3. 두 결과 병합
model_compare <- bind_rows(ggm_data, gm_data)

# 4. 90세 이상 구간에서 RMSE, MAE, MAPE 계산 함수
compute_metrics_85plus <- function(df) {
  df %>%
    filter(.data$age >= 90) %>%
    group_by(year, model) %>%
    summarise(
      RMSE = rmse(mu_obs, mu_pred),
      MAE  = mae(mu_obs, mu_pred),
      MAPE = mean(abs((mu_obs - mu_pred) / mu_obs)) * 100,
      .groups = "drop"
    ) %>%
    arrange(year, model)
}

# 5. 계산 실행
metrics_90plus <- compute_metrics_85plus(model_compare) %>%
  filter(year >= 1990, year <= 2020)

# 6. 결과 확인
print(metrics_90plus, n = 100)

# MAPE를 기준으로 GM vs GGM 비교 시각화
ggplot(metrics_90plus, aes(x = year, y = RMSE, color = model, group = model)) +
  geom_line(linewidth = 1.1) +
  geom_point(size = 2) +
  labs(
    title = "90세 이상 사력의 예측 오차 비교 (여성)",
    x = "연도", y = "MAPE (%)"
  ) +
  theme_minimal() +
  scale_color_manual(values = c("GGM" = "steelblue", "GM" = "tomato"))

# ------------------------------------------





# ------------------------------------------
# GM 적합 잘됐는지 평가
library(purrr)
library(tibble)

gm_fit_summary <- male_with_obs %>%
  filter(!is.na(mu_obs), mu_obs > 0) %>%        # 로그변환 가능성 확보
  group_by(year) %>%
  group_map(~{
    data_year <- .x
    yr <- unique(data_year$year)
    
    # 선형회귀: log(mu_obs) ~ age
    fit_lm <- lm(log(mu_obs) ~ age, data = data_year)
    fit_sum <- summary(fit_lm)
    
    tibble(
      year = yr,
      intercept = coef(fit_lm)[1],
      slope     = coef(fit_lm)[2],
      r_squared = fit_sum$r.squared,
      p_value   = coef(fit_sum)[2, 4]
    )
  }) %>%
  bind_rows()

# 결과 출력
print(n= 35, gm_fit_summary)
# ------------------------------------------
