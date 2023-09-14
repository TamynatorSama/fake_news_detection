#libraries importation
import streamlit as st 
import numpy as np
import pickle
import re 
import string 

#reading downloaded stack model
stack_model = pickle.load(open('./stack_model.sav', 'rb'))

#read tfidf model
tfidf_model = pickle.load(open('./tfidf_model.sav', 'rb'))


#texts cleaning
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

    
#interface
def main():
    #build interface
    st.header('FAKE NEWS DETECTION SYSTEM')

    st.markdown('Kindly input news content/text')

    #get user inputs
    text = st.text_area('News')    
    
    #variable for storing output
    output = ''
    
    if st.button('Classify news'):
        #cleaning the text gotten
        clean_text = wordopt(text)
        #convert the cleaned text into numerical data
        content = tfidf_model.transform([clean_text]).toarray()

       
        result = stack_model.predict(content) 
        
        if result == 0:
            output = 'This is a fake news' 
        else:
            output = 'This is a real news'
    
    st.info(output)
    


#launch app
if __name__ == '__main__':
    main()
