###############################################
###### MENGIMPOR LIBRARY YANG DIBUTUHKAN ######
###############################################
import streamlit as st
import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing

#############################################
###### FUNGSI UNTUK TEXT PREPROCESSING ######
#############################################
def wordopt(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text) 
    text = re.sub(r"\W", " ", text)      
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  
    text = re.sub(r'<.*?>+', '', text)  
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  
    text = re.sub(r'\w*\d\w*', '', text)  
    return text

#########################################
###### LOAD DAN PREPROCESSING DATA ######
#########################################
df = pd.read_csv('dataset/news.csv')
df = df.drop(["Unnamed: 0", "title"], axis=1)

le = preprocessing.LabelEncoder()
le.fit(df['label'])
df['label'] = le.transform(df['label'])


#############################
###### SPLITTING DATA ######
############################
x = df['text']
y = df['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


##########################
###### VECTORIZATION #####
##########################
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


#########################
###### TRAIN MODEL #####
########################
LR = LogisticRegression()
LR.fit(xv_train, y_train)

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

GB = GradientBoostingClassifier(random_state=0)
GB.fit(xv_train, y_train)

RF = RandomForestClassifier(random_state=0)
RF.fit(xv_train, y_train)


#################################
###### STREAMLIT APP LAYOUT #####
################################
st.title("Fake News Detector")

input_news = st.text_area("Enter the news text", "", height=500)  # Increased height to make the box larger



########################################
###### PREDIKSI DAN EVALUASI MODEL #####
########################################
if st.button("Classify"):
    if input_news:
        new_def_test = pd.DataFrame({"text": [input_news]})
        new_def_test['text'] = new_def_test['text'].apply(wordopt)
        new_x_test = new_def_test['text']
        new_xv_test = vectorization.transform(new_x_test)

        pred_LR = LR.predict(new_xv_test)
        pred_DT = DT.predict(new_xv_test)
        pred_GB = GB.predict(new_xv_test)
        pred_RF = RF.predict(new_xv_test)

        st.write(f"**Logistic Regression Prediction:** {'Fake News' if pred_LR[0] == 0 else 'Not Fake News'}")
        st.write(f"**Decision Tree Prediction:** {'Fake News' if pred_DT[0] == 0 else 'Not Fake News'}")
        st.write(f"**Gradient Boosting Prediction:** {'Fake News' if pred_GB[0] == 0 else 'Not Fake News'}")
        st.write(f"**Random Forest Prediction:** {'Fake News' if pred_RF[0] == 0 else 'Not Fake News'}")

        # Classification report for Logistic Regression
        st.write("### Logistic Regression Model Evaluation")
        pred_lr = LR.predict(xv_test)
        report = classification_report(y_test, pred_lr, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
        
        # Classification report for Decision Tree
        st.write("### Decision Tree Model Evaluation")
        pred_dt = DT.predict(xv_test)
        report = classification_report(y_test, pred_dt, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        # Classification report for Gradient Boosting
        st.write("### Gradient Boosting Model Evaluation")
        pred_gb = GB.predict(xv_test)
        report = classification_report(y_test, pred_gb, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        # Classification report for Random Forest
        st.write("### Random Forest Model Evaluation")
        pred_rf = RF.predict(xv_test)
        report = classification_report(y_test, pred_rf, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
