import pandas as pd
import matplotlib.pyplot as plt
import ydata_profiling as yp
from dataHandle import datasetaq
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

# ---------------------------- EDA ----------------------------- #

#Datafdrame creation
df = datasetaq()

#Series definition
ger = df['val_geracao']
con = df['val_cargaenergiamwmed']

#Descriptive stats
profile = yp.ProfileReport(df, title="My Dataset Profile")
# profile.to_file("profile_report.html")

#Boxplot
df.boxplot()

#Scatter plot generation x consumption
df.plot.scatter(x='val_geracao', y='val_cargaenergiamwmed')

#Generation and Consumptionm ACF and PACF
fig, ax,  = plt.subplots(2, 1, figsize=(6, 6))
plot_acf(ger, ax=ax[0], lags=40)
ax[0].set_title("Generation ACF Plot")
plot_pacf(ger, ax=ax[1], lags=40, method='ywm')
ax[1].set_title("Generation PACF Plot")

#Consumption ACF and PACF
fig2, ax2,  = plt.subplots(2, 1, figsize=(6, 6))
plot_acf(con, ax=ax2[0], lags=40)
ax2[0].set_title("Consumption ACF Plot")
plot_pacf(con, ax=ax2[1], lags=40, method='ywm')
ax2[1].set_title("Consumption PACF Plot")

#Grouping figs
plt.tight_layout()

#Show plots
plt.show()




