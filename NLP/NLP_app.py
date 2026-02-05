import streamlit as st 
import nltk
from nltk.tokenize import (word_tokenize,sent_tokenize,WhitespaceTokenizer,
                           BlanklineTokenizer,WordPunctTokenizer)

from nltk.stem import (PorterStemmer,LancasterStemmer,SnowballStemmer,
                       WordNetLemmatizer)
from nltk.corpus import stopwords
from nltk import ne_chunk
from nltk.tree import Tree
from nltk import pos_tag
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

st.set_page_config(page_title = "NLU using NLTK", layout="wide")
st.title("🧠 NLU using NLTK ")

text = st.text_area("✍️ Enter a paragraph:", height=200)

if st.button("Analyze Text") and text.strip():
    
     # ---------------- Sentence Tokenization ----------------
     sentences = sent_tokenize(text)
     st.subheader("🔹 Sentence Tokenization")
     st.write(sentences)
     st.success(f"Number of sentences: {len(sentences)}")
     
     # ---------------- Word Tokenization --------------------
     words = word_tokenize(text)
     st.subheader("🔹 Word Tokenization")
     st.write(words)
     st.success(f"Number of words: {len(words)}")
     
     # ---------------- Whitespace Tokenization ----------------
     ws_tokens = WhitespaceTokenizer().tokenize(text)
     st.subheader("🔹 Whitespace Tokenization")
     st.write(ws_tokens)
     st.success(f"Whitesace tokens count: {len(ws_tokens)}")
     
     # ---------------- Blank Line Tokenization ----------------
     bl_tokens = BlanklineTokenizer().tokenize(text)
     st.subheader("🔹 Blank Line Tokenization")
     st.write(bl_tokens)
     st.success(f"BlankLine tokens count: {len(bl_tokens)}")
     
     # ---------------- WordPunct Tokenization ----------------
     wp_tokens = WordPunctTokenizer().tokenize(text)
     st.subheader("🔹 WordPunct Tokenization")
     st.write(wp_tokens)
     st.success(f"WordPunct tokens count: {len(wp_tokens)}")
     
     # ---------------- Stopwords Removal ----------------
     stop_words = set(stopwords.words("english"))
     filtered_words = [w for w in words if w.lower() not in stop_words and w.isalpha()]
     st.subheader("🚫 Stopwords Removal")
     st.write(filtered_words)
     
     # ---------------- POS Tagging ----------------
     st.subheader("🏷️ POS Tagging (Part of Speech)")

     pos_tags = pos_tag(words)
     st.write(pos_tags)

     st.success("POS tagging assigns grammatical roles like noun, verb, adjective, etc.")
     
     # ---------------- Named Entity Recognition ----------------
     st.subheader("🧠 Named Entity Recognition (NER)")

     ner_tree = ne_chunk(pos_tags)

     entities = []
     for subtree in ner_tree:
         if isinstance(subtree, Tree):
            entity = " ".join([token for token, pos in subtree.leaves()])
            label = subtree.label()
            entities.append((entity, label))

     if entities:
         st.write(entities)
     else:
         st.info("No named entities detected.")
         
     st.subheader("🌳 NER Tree Visualization")
     st.write(ner_tree)    
     
     # ---------------- Porter Stemmer ----------------
     porter = PorterStemmer()
     porter_stems = [porter.stem(word) for word in words]
     st.subheader("🔹 Porter Stemmer")
     st.write(wp_tokens)
     st.success(f"Porter Stemmer words count: {len(porter_stems)}")
     
     # ---------------- Lancaster Stemmer ----------------
     lancaster = LancasterStemmer()
     lancaster_stems = [lancaster.stem(word) for word in words]
     st.subheader("🔹 Lancaster Stemmer")
     st.write(lancaster_stems)
     st.success(f"Lancaster Stemmer words count: {len(lancaster_stems)}")
     
     # ---------------- Snowball Stemmer ----------------
     snowball = SnowballStemmer("english")
     snowball_stems = [snowball.stem(word) for word in words]
     st.subheader("🔹 Snowball Stemmer")
     st.write(snowball_stems)
     st.success(f"Snowball stemmmed words count: {len(snowball_stems)}")
     
     # ---------------- Lemmatization ----------------
     lemmatizer = WordNetLemmatizer()
     lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
     st.subheader("🔹 Lemmatization")
     st.write(lemmatized_words)
     st.success(f"Lemmatized words count: {len(lemmatized_words)}")
     
     #---------------- Bag of Words ----------------
     st.subheader("📊 Bag of Words (BoW)")
     bow = CountVectorizer()
     bow_matrix = bow.fit_transform([text])
     st.write(dict(zip(bow.get_feature_names_out(), bow_matrix.toarray()[0])))

    # ---------------- TF-IDF ----------------
     st.subheader("📈 TF-IDF")
     tfidf = TfidfVectorizer()
     tfidf_matrix = tfidf.fit_transform([text])
     st.write(dict(zip(tfidf.get_feature_names_out(), tfidf_matrix.toarray()[0])))

    # ---------------- Word2Vec ----------------
     st.subheader("🧬 Word2Vec Embeddings")
     tokenized_sentences = [word_tokenize(sent.lower()) for sent in sentences]

     w2v_model = Word2Vec(sentences=tokenized_sentences,vector_size=50,
                      window=3,min_count=1,workers=4 )

     sample_word = tokenized_sentences[0][0]
     st.write(f"Vector for word **'{sample_word}'**:")
     st.write(w2v_model.wv[sample_word])
     
     # ---------------- WordCloud ----------------
     st.subheader("☁️ WordCloud")
     wordcloud = WordCloud(width=600, height=300, background_color="lightblue").generate(text)
     
     fig,ax = plt.subplots(figsize=(10,5))
     ax.imshow(wordcloud,interpolation="bilinear")
     ax.axis("off")
     st.pyplot(fig)
     
     st.success(f"Unique words in WordCloud: {len(set(words))}")
     
else:
    st.info("👆 Please enter a paragraph and click ** Analyze Text **")
              
     