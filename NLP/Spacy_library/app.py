import streamlit as st
import spacy
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
import pdfplumber
from io import StringIO
from collections import Counter
import math

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Advanced NLP Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ---------------- Load spaCy ----------------
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()

# ---------------- Header ----------------
st.markdown("""
<h1 style='text-align:center;'>🧠 Advanced NLP Analyzer</h1>
<p style='text-align:center;color:gray;'>
Text Analysis • Summarization • Visualization (spaCy + Streamlit)
</p>
""", unsafe_allow_html=True)

st.divider()

# ---------------- File Upload (MAIN VIEW) ----------------
uploaded_file = st.file_uploader(
    "📂 Upload TXT or PDF (optional)",
    type=["txt", "pdf"]
)

text = ""

if uploaded_file:
    if uploaded_file.type == "text/plain":
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        text = stringio.read()
    else:
        with pdfplumber.open(uploaded_file) as pdf:
            text = "\n".join(
                [page.extract_text() for page in pdf.pages if page.extract_text()]
            )

# ---------------- Text Input ----------------
text = st.text_area(
    "✍️ Enter text for NLP analysis",
    value=text,
    height=200,
    placeholder="Paste text here or upload a document above..."
)

if not text.strip():
    st.info("👆 Enter or upload text to start NLP analysis")
    st.stop()

# ---------------- NLP Processing ----------------
doc = nlp(text)

# ---------------- Metrics ----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tokens", len(doc))
c2.metric("Sentences", len(list(doc.sents)))
c3.metric("Entities", len(doc.ents))
c4.metric("POS Tags", len(set([t.pos_ for t in doc])))

st.divider()

# ---------------- Text Summarization ----------------
st.subheader("🧠 Text Summarization")

# Word frequency (excluding stopwords & punctuation)
word_freq = Counter(
    token.text.lower()
    for token in doc
    if token.is_alpha and not token.is_stop
)

max_freq = max(word_freq.values()) if word_freq else 1
for word in word_freq:
    word_freq[word] /= max_freq

# Sentence scoring
sent_scores = {}
for sent in doc.sents:
    for word in sent:
        if word.text.lower() in word_freq:
            sent_scores[sent] = sent_scores.get(sent, 0) + word_freq[word.text.lower()]

# Select top sentences
summary_length = max(1, math.ceil(len(sent_scores) * 0.3))
summary_sentences = sorted(
    sent_scores, key=sent_scores.get, reverse=True
)[:summary_length]

summary = " ".join([sent.text for sent in summary_sentences])

st.success(summary)

st.divider()

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📈 Scores",
    "☁️ WordCloud",
    "🏷️ Entities",
    "🌳 Dependency Tree"
])

# ---------------- Overview ----------------
with tab1:
    st.subheader("📌 Original Text")
    st.write(text)

    sentiment = TextBlob(text).sentiment.polarity
    if sentiment > 0:
        st.success(f"😊 Positive Sentiment ({sentiment:.2f})")
    elif sentiment < 0:
        st.error(f"😞 Negative Sentiment ({sentiment:.2f})")
    else:
        st.warning("😐 Neutral Sentiment")

# ---------------- Token & Sentence Scores ----------------
with tab2:
    st.subheader("📊 Token Frequency")

    token_scores = {
        token.text: token_scores + 1
        for token in doc
        if token.is_alpha and not token.is_stop
        for token_scores in [0]
    }

    df_token = pd.DataFrame(
        Counter(token_scores).most_common(10),
        columns=["Token", "Score"]
    )

    fig1, ax1 = plt.subplots()
    ax1.bar(df_token["Token"], df_token["Score"])
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    st.subheader("📊 Sentence Length Scores")
    sent_len = [len([t for t in sent if t.is_alpha]) for sent in doc.sents]

    fig2, ax2 = plt.subplots()
    ax2.bar(range(len(sent_len)), sent_len)
    ax2.set_xlabel("Sentence Index")
    ax2.set_ylabel("Token Count")
    st.pyplot(fig2)

# ---------------- WordCloud ----------------
with tab3:
    st.subheader("☁️ WordCloud")

    wc = WordCloud(
        width=900,
        height=400,
        background_color="white"
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

# ---------------- Entities ----------------
with tab4:
    if doc.ents:
        ent_data = [{"Entity": ent.text, "Label": ent.label_} for ent in doc.ents]
        st.dataframe(pd.DataFrame(ent_data), use_container_width=True)

        st.markdown(
            spacy.displacy.render(doc, style="ent"),
            unsafe_allow_html=True
        )
    else:
        st.warning("No named entities found")

# ---------------- Dependency Tree ----------------
with tab5:
    st.markdown(
        spacy.displacy.render(doc, style="dep"),
        unsafe_allow_html=True
    )

# ---------------- Footer ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;'>Built with ❤️ using spaCy & Streamlit</p>",
    unsafe_allow_html=True
)
