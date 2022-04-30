# %%
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import func
import numpy as np

plt.style.use(['science', 'no-latex', 'grid'])

data = pd.read_csv('../results/MonthlyPrediction/Historicaldata.csv')

plt.scatter(data["PrMean"].values, data["TempMean"].values,  s=5)
plt.xlim(0, 5)
plt.ylim(-10, 20)
plt.xlabel("Precipitation(mm)")
plt.ylabel("Temperature (Celcius)")
plt.savefig("../results/MonthlyPrediction/PreTempCohort.tiff",
            dpi=300, bbox_inches='tight')
plt.show()


# Hello world
# %%
