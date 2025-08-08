import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle

# Title

col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']

st.title('California Housing Price Prediction')

st.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTfsUfHqe99yq7NEanwMQpm9JD5Vv2xNChiEHveduq8wb46gU3lhef5tcXpitfKrDFqWlI&usqp=CAU')
st.header('Model of housing prices to predict median house values in California',divider = True)

#st.subheader('''User Must Enter Given Values to Predict Price:
#['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')

st.sidebar.title('Select House Features 🏠')

st.sidebar.image('https://png.pngtree.com/thumb_back/fh260/background/20240614/pngtree-modern-luxurious-real-estate-property-house-background-image_15868464.jpg')

# read data
temp_df = pd.read_csv('california.csv')
random.seed(20)

all_values =[]

for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg(['min','max'])

    var = st.sidebar.slider(f'Select {i} value', int(min_value), int(max_value),
                      random.randint(int(min_value),int(max_value)))

    all_values.append(var)

ss = StandardScaler()
ss.fit(temp_df[col])

final_value = ss.transform([all_values])


with open('Property_Price_Prediction_Project.pkl','rb') as f:
    chatgpt = pickle.load(f)


price = chatgpt.predict(final_value)[0]


import time


st.write(pd.DataFrame(dict(zip(col,all_values)),index = [1]))


progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.write('Predicted Price')
place = st.empty()
place.image('https://cdn-icons-gif.flaticon.com/11677/11677497.gif',width=80)

if price>0:
    for i in range(100):
        time.sleep(0.02)
        progress_bar.progress(i+1)
        
    body = f'Predicted Median House Price:, ${round(price,2)}Thousand Dollar'
    placeholder.empty()
    
    st.success(body)
else:
    body = 'Invalid House Values'
    st.warning(body)
