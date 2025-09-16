from matplotlib.ticker import MultipleLocator
import func
import matplotlib.pyplot as plt

years, ages, df_mu, df_Dx, df_Ex = func.load_life_table(key = "kr", sex = "여자")

for year in range(2020, 2024):
    plt.plot(df_mu.loc[80:].index, df_mu.loc[80:, str(year)])
plt.gca().xaxis.set_major_locator(MultipleLocator(1))  # 1 단위로 눈금    
plt.legend()
plt.show()

