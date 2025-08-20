import streamlit as st
import pickle

# Load trained model and vectorizer
with open('svm_spam_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

st.title("📧 Email / SMS Spam Detection with Confidence")
st.write("Enter a message and see if it's Spam or Ham, with model confidence!")

# Input message
message = st.text_area("Enter your message:")

if st.button("Predict"):
    if message.strip() == "":
        st.warning("Please enter a message to predict.")
    else:
        # Transform message and predict
        message_vec = vectorizer.transform([message])
        prediction = svm_model.predict(message_vec)[0]
        prob = svm_model.predict_proba(message_vec)[0]

        # Display results
        result = "Spam ❌" if prediction == 1 else "Ham ✅"
        st.success(f"Prediction: {result}")
        st.info(f"Confidence: Ham = {prob[0]*100:.2f}%, Spam = {prob[1]*100:.2f}%")
