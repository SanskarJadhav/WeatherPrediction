import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from datetime import date, datetime, timedelta
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error

dfairport = pd.read_csv("indianairports.csv")
nowtime = datetime.now().strftime("%Y-%m-%dT%H:00:00")

# Using VisualCrossing Weather API (1000 free records per day)
def dataretrieval(place):
  lastdate = str(date.today())
  firstdate = str(date.today() - timedelta(days=3))
  url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{0}/{1}/{2}?unitGroup=metric&include=hours&key=9JHWDJKQMNEK2NZP38WFKRWRA&contentType=csv".format(place, firstdate, lastdate)
  url = url.replace(" ", "%20")
  df = pd.read_csv(url)
  return df

def createdata(dataset, behind):
    X, Y = [], []
    for i in range(len(dataset)-behind):
        a = dataset[i:(i+behind), 0]
        X.append(a)
        Y.append(dataset[i + behind, 0])
    return np.array(X), np.array(Y)

def LSTMimplementation(df):
  df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%dT%H:%M:%S')
  df['day'] = df['datetime'].apply(lambda x: x.day)
  df['hour'] = df['datetime'].apply(lambda x: x.hour)
  row_num = df[df['datetime'] == str(nowtime)].index
  features = ['temp', 'humidity', 'windgust', 'winddir']
  imp_array = []
  predicttoday = []
  for i in features:
    arr = []
    df_update = df.loc[:-24,['datetime',i, 'day', 'hour']]
    dataset = df_update[i].values
    dataset = np.reshape(dataset, (-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # apart from yesterday, all other days can be used to train model
    train_size = len(dataset) - 29
    train = dataset[:train_size,:]
    test = dataset[-29:,:]
    # using past 5 records to predict weather in next hour
    X_train, Y_train = createdata(train, 5)
    X_test, Y_test = createdata(test, 5)
    # reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    model = Sequential()
    model.add(LSTM(256, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(X_train, Y_train, epochs=15, batch_size=8, validation_data=(X_test, Y_test), 
                        callbacks=[EarlyStopping(monitor='val_loss', patience=5)], verbose=1, shuffle=False)
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    # revert predictions
    train_predict = scaler.inverse_transform(train_predict)
    Y_train = scaler.inverse_transform([Y_train])
    test_predict = scaler.inverse_transform(test_predict)
    Y_test = scaler.inverse_transform([Y_test])
    arr.append(f'Train MAE for {i}: {mean_absolute_error(Y_train[0], train_predict[:,0])}')
    arr.append(f'Train RMSE for {i}: {np.sqrt(mean_squared_error(Y_train[0], train_predict[:,0]))}')
    arr.append(f'Test MAE for {i}: {mean_absolute_error(Y_test[0], test_predict[:,0])}')
    arr.append(f'Test RMSE for {i}: {np.sqrt(mean_squared_error(Y_test[0], test_predict[:,0]))}')
    fig = plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    arr.append(fig)
    aa=[x for x in range(Y_test.shape[1])]
    fig1 = plt.figure(figsize=(8,4))
    plt.plot(aa, Y_test[0][:], color='blue', marker='.', label="actual")
    plt.plot(aa, test_predict[:,0][:], color='purple', marker='s', label="predicted")
    plt.ylabel(i, size=12)
    plt.xlabel('Hours', size=12)
    plt.legend(fontsize=9)
    arr.append(fig1)
    imp_array.append(arr)
    dftoday = df.loc[(row_num-4):(row_num+1),['datetime',i, 'day', 'hour']]
    datatoday = dftoday[i].values
    datatoday = np.reshape(datatoday, (-1, 1))
    datatoday = scaler.fit_transform(datatoday)
    predtoday = model.predict(datatoday)
    predtoday = scaler.inverse_transform(predtoday)
    predicttoday.append([i, predtoday])
  return imp_array, predicttoday



def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.pixabay.com/photo/2018/08/23/07/35/thunderstorm-3625405_1280.jpg");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Streamlit Website Deployment
st.set_page_config(layout='wide', page_title="Weather Oracle")
add_bg_from_url()
tk = 0
st.title("Predict Today's Weather :sunny:")
location = st.selectbox('Airport: ', dfairport['Display Name'], index = 0)
if st.button('Submit'):
    tk = 1
if tk == 1:
    place = dfairport[dfairport['Display Name'] == location]['municipality'].values[0]
    dfweather = dataretrieval(place)
    x, y = LSTMimplementation(dfweather)
    st.header("Prediction for next hour today ({})".format(str(nowtime + timedelta(hours=1))))
    for e in y:
      st.write("{0}: {1}".format(e[0], e[1]))
    st.header("Prediction vs. Actual Comparison for Yesterday")
    for i in x:
        for j in i[:-2]:
            st.write(j)
        st.pyplot(i[-2])
        st.pyplot(i[-1])
