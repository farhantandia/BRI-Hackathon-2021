import numpy as np
import pandas as pd

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Forecasting
from fbprophet import Prophet
# from fbprophet.plot import plot_plotly, plot_components_plotly
# import plotly.graph_objs as go
cash = pd.read_csv('train.csv')
cash.head()

cash['periode'] = pd.to_datetime(cash['periode'])
cash.sort_values('periode', inplace=True)
cash['kas_kantor_verif'] = cash['kas_kantor'].shift()+cash['cash_in_kantor'] + cash['cash_out_kantor']
cash['kas_echannel_verif'] = cash['kas_echannel'].shift()+cash['cash_in_echannel'] + cash['cash_out_echannel']

cash.set_index('periode')[['kas_kantor', 'kas_echannel']].plot(subplots=True,
                                                                      figsize=(15, 10))
plt.suptitle('DAILY CASH RATIO:')
# plt.show()

daily_kas_kantor = cash[['periode', 'kas_kantor']].rename(
    columns={'periode': 'ds',
             'kas_kantor': 'y'})

# Take 80% data for training

# daily_kas_kantor = cash[['periode', 'kas_kantor']].head(int(len(cash)*(80/100))).rename(
#     columns={'periode': 'ds',
#              'kas_kantor': 'y'})

daily_kas_kantor.head()

daily_kas_echannel = cash[['periode', 'kas_echannel']].rename(
    columns={'periode': 'ds',
             'kas_echannel': 'y'})
# daily_kas_echannel = cash[['periode', 'kas_echannel']].head(int(len(cash)*(80/100))).rename(
#     columns={'periode': 'ds',
#              'kas_echannel': 'y'})

print(daily_kas_echannel.head())

Covid = pd.DataFrame({
    'holiday': 'Covid',
    'ds': pd.to_datetime(['2020-05-19']), # future date, to be forecasted
    'lower_window': -1, # include 27th - 31st December
    'upper_window': 1})

Kemerdekaan = pd.DataFrame({
    'holiday': 'Kemerdekaan',
    'ds': pd.to_datetime(['2020-08-17']), # future date, to be forecasted
    'lower_window': -1, # include 27th - 31st December
    'upper_window': 1})

Natal = pd.DataFrame({
    'holiday': 'Natal',
    'ds': pd.to_datetime(['2020-12-25', # past date, historical data 
                          '2021-12-25']), # future date, to be forecasted
    'lower_window': -10, # include 27th - 31st December
    'upper_window': 0})

holiday_kantor = pd.concat([Covid,Kemerdekaan,Natal], axis=0).reset_index(drop=True)
print(holiday_kantor)

model_holiday_indo_kantor = Prophet(weekly_seasonality=3,
                                    yearly_seasonality=10,
#                                       holidays=holiday_kantor,
                                      changepoint_range=0.8, # default = 0.8 Recommended range: [0.8, 0.95]
                                      changepoint_prior_scale=0.095 # default =0.05 Recommended range: [0.001, 0.5]
                                     )
model_holiday_indo_kantor.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model_holiday_indo_kantor.add_country_holidays(country_name='ID')
model_holiday_indo_kantor.fit(daily_kas_kantor)

# forecasting
future_kantor = model_holiday_indo_kantor.make_future_dataframe(periods=31, freq='D')
forecast_kantor = model_holiday_indo_kantor.predict(future_kantor)

# visualize
model_holiday_indo_kantor.train_holiday_names

# plot_plotly(model_holiday_indo_kantor, forecast_kantor)
# plot_components_plotly(model_holiday_indo_kantor, forecast_kantor)

from sklearn.metrics import mean_squared_log_error
err= np.sqrt(mean_squared_log_error(forecast_kantor['yhat'].head(425),daily_kas_kantor.loc[:,'y']))
print('log mse:',err)


