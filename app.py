import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report
from textblob import TextBlob
import string
import pickle
import streamlit as st

pickle_in = open("fb_LRModel.pkl","rb")
Model = pickle.load(pickle_in)
                 
def lemmat(text):
    lemma=WordNetLemmatizer()
    words=word_tokenize(text)
    return ' '.join([lemma.lemmatize(word) for word in words])

def getsubjectivity(text):
    return TextBlob(text).sentiment.subjectivity
def getpolarity(text):
    return TextBlob(text).sentiment.polarity

def char_rmvl(text):               
    new=[char for char in text if char not in string.punctuation]
    vl=''.join(new)
    new.clear()
    return vl
                 
def preprocess(Headlines):
    Headlines1 = [ x.lower() for x in Headlines]
    Headlines1 = [char_rmvl(x) for x in Headlines1]
    stop = stopwords.words('english')
    Headlines1 = [' '.join([word for word in s.split() if word not in (stop)]) for s in Headlines1]
    Headlines1 = [lemmat(s) for s in Headlines1]
    return Headlines1

def price(High,Low,Open,Volume,Headlines):
    ml = pd.DataFrame()
    ml['Volume']=Volume,Volume
    ml['Open']=Open,Open
    ml['High']=High,High
    ml['Low']=Low,Low
    ml['Headlines']=Headlines,Headlines
    ml['Headlines']=ml['Headlines'].astype(str)
    sid= SentimentIntensityAnalyzer()
    ml['compound'] = ml['Headlines'].apply(lambda x: sid.polarity_scores(x)['compound'])
    ml['negative'] = ml['Headlines'].apply(lambda x: sid.polarity_scores(x)['neg'])
    ml['neutral'] = ml['Headlines'].apply(lambda x: sid.polarity_scores(x)['neu'])
    ml['positive'] = ml['Headlines'].apply(lambda x: sid.polarity_scores(x)['pos'])

    ml['Subjectivity']=ml['Headlines'].apply(getsubjectivity)
    ml['Polarity']=ml['Headlines'].apply(getpolarity)

    k= ml[['Open','High','Low','Volume','positive','neutral','positive','compound','Subjectivity','Polarity']]
    return k

def main():
    st.title("Facebook Inc. Stock price Pridiction")
    html_temp = """
    <div style="background-color:rgb(128, 0, 255);padding:10px">
    <h2 style="color:rgb(255, 124, 37);text-shadow: 0 4px 10px rgba(0, 0, 0, 0.603);text-align:center;">Facebook Inc.Predicted Closed Price</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    High = st.text_input("High")
    Low = st.text_input("Low")
    Open = st.text_input("Open")
    Volume = st.text_input("Volume")
    Headlines= st.text_input("Headlines")
    Headlines=list(Headlines.split("-"))
    hdlines1 = preprocess(Headlines)
    kn=pd.DataFrame()
    kn=price(High,Low,Open,Volume,hdlines1)
    kn1=np.array(kn)
   
    
    ans=''
    if st.button("Predict"):
        ans = Model.predict(kn1)[0]
    st.success('Predicted Price : $ {}'.format(ans))            
if __name__ =='__main__':
    main()
