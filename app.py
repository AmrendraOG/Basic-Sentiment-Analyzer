import streamlit as st
import string
from collections import Counter
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def analyze_text(text, emotion_file='emotions.txt'):
    lower_case = text.lower()
    cleaned_text = lower_case.translate(str.maketrans('', '', string.punctuation))
    tokenized_words = word_tokenize(cleaned_text, "english")
    filtered = [word for word in tokenized_words if word not in stopwords.words('english')]

    emotion_list = []
    with open(emotion_file, 'r') as file:
        for line in file:
            clear_line = line.translate(str.maketrans('', '', "'\n,")).strip()
            if ':' in clear_line:
                word, emotion = clear_line.split(':')
                if word in filtered:
                    emotion_list.append(emotion.strip())

    return filtered, emotion_list, cleaned_text


def get_sentiment(text):
    score = SentimentIntensityAnalyzer().polarity_scores(text)
    if score['neg'] > score['pos']:
        return "Negative", score['neg']
    elif score['pos'] > score['neg']:
        return "Positive", score['pos']
    else:
        return "Neutral", score['neu']


st.title("🧠 Emotion and Sentiment Analyzer")

input_text = st.text_area("Paste your text here:", height=200)

if st.button("Analyze"):
    if input_text.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        filtered_words, emotions, cleaned_text = analyze_text(input_text)
        sentiment, score = get_sentiment(cleaned_text)

        st.subheader("📊 Sentiment Result")
        st.write(f"**Sentiment:** {sentiment}")
        st.write(f"**Score:** {score:.2f}")

        st.subheader("🎭 Detected Emotions")
        if emotions:
            emotion_count = Counter(emotions)
            st.write(emotion_count)

            fig, ax = plt.subplots()
            ax.bar(emotion_count.keys(), emotion_count.values())
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.write("No emotions found in the text.")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit")
