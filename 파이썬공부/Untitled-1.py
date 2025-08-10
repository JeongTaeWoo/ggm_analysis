import numpy as np

data = np.random.rand(100000)
print(f"평균: {np.mean(data)}")
print(f"표준편차: {np.std(data)}")
print(f"제 1 사분위수: {np.quantile(data, .25)}")
print(f"제 3 사분위수: {np.quantile(data, .75)}")