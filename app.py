# import streamlit as st
# import pickle
# import string
# import nltk

# # Load NLTK stopwords
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))

# # Load trained model and vectorizer
# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("vectorizer.pkl", "rb") as f:
#     vectorizer = pickle.load(f)

# # Text cleaning function
# def clean_text(text):
#     text = text.lower()
#     text = ''.join([ch for ch in text if ch not in string.punctuation])
#     words = text.split()
#     words = [w for w in words if w not in stop_words]
#     return ' '.join(words)

# # Streamlit interface
# st.title("📰 Fake News Detection App")
# st.subheader("Paste a news article or paragraph to detect its authenticity.")

# input_text = st.text_area("Enter News Text Here:", height=200)

# if st.button("Check"):
#     cleaned = clean_text(input_text)
#     vec = vectorizer.transform([cleaned]).toarray()
#     prediction = model.predict(vec)[0]
#     confidence = model.predict_proba(vec)[0].max() * 100

#     if prediction == 1:
#         st.error(f"🔴 Fake News Detected! ({confidence:.2f}% confidence)")
#     else:
#         st.success(f"🟢 Real News! ({confidence:.2f}% confidence)")




# import streamlit as st
# import pandas as pd
# import string
# import nltk
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from nltk.corpus import stopwords

# # Download stopwords
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))

# # Text cleaning function
# def clean_text(text):
#     text = text.lower()
#     text = ''.join([ch for ch in text if ch not in string.punctuation])
#     words = text.split()
#     words = [w for w in words if w not in stop_words]
#     return ' '.join(words)

# # UI Title
# st.title("📰 Fake News Detection App (Train Your Own Model!)")

# # Section 1: Upload or Paste Data
# st.subheader("📂 Upload a Dataset or Paste Your Own Data")

# uploaded_file = st.file_uploader("Upload CSV file with 'text' and 'label' columns", type="csv")
# paste_mode = st.checkbox("Or paste data manually")

# df = None

# if paste_mode:
#     pasted_data = st.text_area("Paste your data below (format: text,label):", height=150)
#     if pasted_data:
#         try:
#             lines = pasted_data.strip().split('\n')
#             data = [line.split(',', 1) for line in lines if ',' in line]
#             df = pd.DataFrame(data, columns=["text", "label"])
#             df['label'] = df['label'].astype(int)
#         except Exception as e:
#             st.error(f"Error parsing pasted data: {e}")
# elif uploaded_file:
#     try:
#         df = pd.read_csv(uploaded_file)
#         if "text" not in df.columns or "label" not in df.columns:
#             st.error("CSV must contain 'text' and 'label' columns.")
#             df = None
#     except Exception as e:
#         st.error(f"Error reading CSV: {e}")

# # Section 2: Train Model
# if df is not None:
#     st.write("📊 Preview of Your Data:")
#     st.dataframe(df.head())

#     if st.button("🧠 Train Model"):
#         try:
#             # df["cleaned_text"] = df["text"].apply(clean_text)
#             df["cleaned_text"] = df["text"].apply(clean_text)
#             df = df[df["cleaned_text"].str.strip() != ""]  # remove empty cleaned rows
#             if len(df) < 10:
#                 st.warning("Training with very few samples may produce poor results.")
#             tfidf = TfidfVectorizer(max_features=5000)
#             X = tfidf.fit_transform(df["cleaned_text"]).toarray()
#             y = df["label"]

#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#             model = LogisticRegression()
#             model.fit(X_train, y_train)

#             acc = accuracy_score(y_test, model.predict(X_test))
#             st.success(f"✅ Model trained successfully! Accuracy: {acc:.2f}")

#             # Store model and vectorizer
#             st.session_state['model'] = model
#             st.session_state['vectorizer'] = tfidf
#         except Exception as e:
#             st.error(f"❌ Error during training: {e}")

# # Section 3: Predict Input News
# st.subheader("🧪 Test News Authenticity")

# news_input = st.text_area("Enter a news article to test:", height=200)

# if st.button("Check News"):
#     if 'model' in st.session_state and 'vectorizer' in st.session_state:
#         cleaned = clean_text(news_input)
#         vec = st.session_state['vectorizer'].transform([cleaned]).toarray()
#         pred = st.session_state['model'].predict(vec)[0]
#         conf = st.session_state['model'].predict_proba(vec)[0].max() * 100
#         if pred == 1:
#             st.error(f"🔴 Fake News Detected! ({conf:.2f}% confidence)")
#         else:
#             st.success(f"🟢 Real News! ({conf:.2f}% confidence)")
#     else:
#         st.warning("⚠️ Please upload data and train the model first.")



# import streamlit as st
# import pickle
# import string
# import nltk
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# from nltk.corpus import stopwords
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import pandas as pd

# # Load NLTK stopwords
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))

# # Load trained model and vectorizer
# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("vectorizer.pkl", "rb") as f:
#     vectorizer = pickle.load(f)

# # Text cleaning function
# def clean_text(text):
#     text = text.lower()
#     text = ''.join([ch for ch in text if ch not in string.punctuation])
#     words = text.split()
#     words = [w for w in words if w not in stop_words]
#     return ' '.join(words)

# # Store previous news in a list
# previous_news = []

# # Compare entered news with previous news
# def compare_with_previous_news(new_text, previous_news_list):
#     new_vec = vectorizer.transform([new_text]).toarray()  # Transform new text
#     similarities = []
#     for prev_news in previous_news_list:
#         prev_vec = vectorizer.transform([prev_news]).toarray()  # Transform previous news
#         sim = cosine_similarity(new_vec, prev_vec)[0][0]  # Cosine similarity score
#         similarities.append(sim)
    
#     max_similarity = max(similarities) if similarities else 0
#     return max_similarity > 0.8  # Threshold for similarity

# # Streamlit interface
# st.title("📰 Fake News Detection App")
# st.subheader("Paste a news article to detect its authenticity.")

# input_text = st.text_area("Enter News Text Here:", height=200)

# if st.button("Check"):
#     if input_text.strip() == "":
#         st.error("Please enter some text.")
#     else:
#         # Clean and predict the entered news
#         cleaned = clean_text(input_text)
#         vec = vectorizer.transform([cleaned]).toarray()  # Transform the input text
#         prediction = model.predict(vec)[0]  # Predict the news class
#         confidence = model.predict_proba(vec)[0].max() * 100  # Get prediction confidence

#         # Check similarity with previous news
#         is_similar = compare_with_previous_news(cleaned, previous_news)

#         if is_similar:
#             st.info("This news is similar to previously entered news.")

#         # Show prediction result
#         if prediction == 1:
#             st.error(f"🔴 Fake News Detected! ({confidence:.2f}% confidence)")
#         else:
#             st.success(f"🟢 Real News! ({confidence:.2f}% confidence)")

#         # Add current news to previous news
#         previous_news.append(cleaned)

# # Allow the user to upload files for training (optional)
# uploaded_file = st.file_uploader("Upload a CSV file for training (optional)", type="csv")
# if uploaded_file:
#     try:
#         df = pd.read_csv(uploaded_file)
#         df["cleaned_text"] = df["text"].apply(clean_text)

#         if df["cleaned_text"].str.strip().eq("").any():
#             st.error("Error during training: empty vocabulary; perhaps the documents only contain stop words.")
#         else:
#             tfidf = TfidfVectorizer(max_features=5000)
#             X = tfidf.fit_transform(df["cleaned_text"]).toarray()
#             y = df["label"]

#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#             model = LogisticRegression()
#             model.fit(X_train, y_train)

#             acc = accuracy_score(y_test, model.predict(X_test))
#             st.success(f"✅ Model trained successfully! Accuracy: {acc:.2f}")

#             # Save new model and vectorizer
#             with open("model.pkl", "wb") as f:
#                 pickle.dump(model, f)
#             with open("vectorizer.pkl", "wb") as f:
#                 pickle.dump(tfidf, f)

#     except Exception as e:
#         st.error(f"Error during training: {e}")




# import streamlit as st
# import pickle
# import string
# import nltk
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer

# # Load NLTK stopwords
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))

# # Load trained model and vectorizer
# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("vectorizer.pkl", "rb") as f:
#     vectorizer = pickle.load(f)

# # Load Sentence Transformer model for similarity comparison
# sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# # Text cleaning function
# def clean_text(text):
#     text = text.lower()
#     text = ''.join([ch for ch in text if ch not in string.punctuation])
#     words = text.split()
#     words = [w for w in words if w not in stop_words]
#     return ' '.join(words)

# # Streamlit interface
# st.title("📰 Fake News Detection and Changed News Detection App")
# st.subheader("Paste a news article to detect its authenticity, and compare it to previous articles.")

# # Step 1: Check if news is real/fake
# input_text = st.text_area("Enter News Text Here:", height=200)

# if st.button("Check Real/Fake"):
#     if input_text.strip() == "":
#         st.error("Please enter some text.")
#     else:
#         # Clean and predict the entered news
#         cleaned = clean_text(input_text)
#         vec = vectorizer.transform([cleaned]).toarray()  # Transform the input text
#         prediction = model.predict(vec)[0]  # Predict the news class
#         confidence = model.predict_proba(vec)[0].max() * 100  # Get prediction confidence

#         if prediction == 1:
#             st.error(f"🔴 Fake News Detected! ({confidence:.2f}% confidence)")
#         else:
#             st.success(f"🟢 Real News! ({confidence:.2f}% confidence)")

# # Step 2: Compare the original and modified news
# input_original_text = st.text_area("Enter Original News Text Here:", height=200)
# input_changed_text = st.text_area("Enter Changed News Text Here:", height=200)

# if st.button("Compare Changed News"):
#     if input_original_text.strip() == "" or input_changed_text.strip() == "":
#         st.error("Please enter both original and changed news text.")
#     else:
#         # Encode the texts using SentenceTransformer
#         original_emb = sentence_model.encode(input_original_text)
#         changed_emb = sentence_model.encode(input_changed_text)

#         # Compute cosine similarity between the original and changed news
#         similarity_score = cosine_similarity([original_emb], [changed_emb])[0][0]

#         st.write(f"Similarity Score: {similarity_score:.2f}")

#         # If the similarity score is above a threshold, consider it similar
#         if similarity_score > 0.8:  # Threshold for similarity
#             st.info("The news has been altered but remains similar to the original.")
#         else:
#             st.warning("The news is significantly different from the original.")



# import difflib
# import streamlit as st
# import pickle
# import string
# import nltk
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer

# # Load NLTK stopwords
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))

# # Load trained model and vectorizer
# with open("model.pkl", "rb") as f:
#     model = pickle.load(f)

# with open("vectorizer.pkl", "rb") as f:
#     vectorizer = pickle.load(f)

# # Load Sentence Transformer model for similarity comparison
# sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# # Text cleaning function
# def clean_text(text):
#     text = text.lower()
#     text = ''.join([ch for ch in text if ch not in string.punctuation])
#     words = text.split()
#     words = [w for w in words if w not in stop_words]
#     return ' '.join(words)

# # Function to compare and highlight changes in the text
# def highlight_changes(original_text, changed_text):
#     # Get the differences using difflib
#     diff = difflib.ndiff(original_text.splitlines(), changed_text.splitlines())
    
#     # Highlight the changes
#     highlighted = ""
#     for line in diff:
#         if line.startswith("-"):  # Deletions in original text
#             highlighted += f'<span style="color:red; text-decoration: line-through;">{line[2:]}</span><br>'
#         elif line.startswith("+"):  # Additions in changed text
#             highlighted += f'<span style="color:green;">{line[2:]}</span><br>'
#         else:  # Unchanged parts
#             highlighted += f'{line[2:]}<br>'
    
#     return highlighted

# # Streamlit interface
# st.title("📰 Fake News Detection and Changed News Detection App")
# st.subheader("Paste a news article to detect its authenticity, and compare it to previous articles.")

# # Step 1: Check if news is real/fake
# input_text = st.text_area("Enter News Text Here:", height=200)

# if st.button("Check Real/Fake"):
#     if input_text.strip() == "":
#         st.error("Please enter some text.")
#     else:
#         # Clean and predict the entered news
#         cleaned = clean_text(input_text)
#         vec = vectorizer.transform([cleaned]).toarray()  # Transform the input text
#         prediction = model.predict(vec)[0]  # Predict the news class
#         confidence = model.predict_proba(vec)[0].max() * 100  # Get prediction confidence

#         if prediction == 1:
#             st.error(f"🔴 Fake News Detected! ({confidence:.2f}% confidence)")
#         else:
#             st.success(f"🟢 Real News! ({confidence:.2f}% confidence)")

# # Step 2: Compare the original and modified news
# input_original_text = st.text_area("Enter Original News Text Here:", height=200)
# input_changed_text = st.text_area("Enter Changed News Text Here:", height=200)

# if st.button("Compare Changed News"):
#     if input_original_text.strip() == "" or input_changed_text.strip() == "":
#         st.error("Please enter both original and changed news text.")
#     else:
#         # Highlight the changes
#         highlighted_changes = highlight_changes(input_original_text, input_changed_text)
        
#         # Display the changes in Streamlit
#         st.markdown(f"**Changes Highlighted:**<br>{highlighted_changes}", unsafe_allow_html=True)

#         # Encode the texts using SentenceTransformer
#         original_emb = sentence_model.encode(input_original_text)
#         changed_emb = sentence_model.encode(input_changed_text)

#         # Compute cosine similarity between the original and changed news
#         similarity_score = cosine_similarity([original_emb], [changed_emb])[0][0]

#         st.write(f"Similarity Score: {similarity_score:.2f}")

#         # If the similarity score is above a threshold, consider it similar
#         if similarity_score > 0.8:  # Threshold for similarity
#             st.info("The news has been altered but remains similar to the original.")
#         else:
#             st.warning("The news is significantly different from the original.")




import streamlit as st
import pickle
import string
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import difflib

# Load NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Load trained model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load Sentence Transformer model for similarity comparison
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = ''.join([ch for ch in text if ch not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

# Function to highlight changes between original and changed text
def highlight_changes(original_text, changed_text):
    # Use difflib to get the differences between the two texts
    matcher = difflib.SequenceMatcher(None, original_text.split(), changed_text.split())
    
    highlighted = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace' or tag == 'insert':  # 'replace' or 'insert' means there's a change
            highlighted.append(f'<span style="background-color: red;">{" ".join(changed_text.split()[j1:j2])}</span>')
        else:
            highlighted.append(" ".join(changed_text.split()[j1:j2]))
    
    return " ".join(highlighted)

# Streamlit interface
st.title("📰 Fake News Detection and Changed News Detection App")
st.subheader("Paste a news article to detect its authenticity, and compare it to previous articles.")

# Step 1: Check if news is real/fake
input_text = st.text_area("Enter News Text Here:", height=200)

if st.button("Check Real/Fake"):
    if input_text.strip() == "":
        st.error("Please enter some text.")
    else:
        # Clean and predict the entered news
        cleaned = clean_text(input_text)
        vec = vectorizer.transform([cleaned]).toarray()  # Transform the input text
        prediction = model.predict(vec)[0]  # Predict the news class
        confidence = model.predict_proba(vec)[0].max() * 100  # Get prediction confidence

        if prediction == 1:
            st.error(f"🔴 Fake News Detected! ({confidence:.2f}% confidence)")
        else:
            st.success(f"🟢 Real News! ({confidence:.2f}% confidence)")

# Step 2: Compare the original and modified news
input_original_text = st.text_area("Enter Original News Text Here:", height=200)
input_changed_text = st.text_area("Enter Changed News Text Here:", height=200)

if st.button("Compare Changed News"):
    if input_original_text.strip() == "" or input_changed_text.strip() == "":
        st.error("Please enter both original and changed news text.")
    else:
        # Highlight the changes using the function
        highlighted_text = highlight_changes(input_original_text, input_changed_text)

        # Display the highlighted differences in HTML
        st.markdown(f"**Highlighted Changes:**<br>{highlighted_text}", unsafe_allow_html=True)

        # Encode the texts using SentenceTransformer for similarity check
        original_emb = sentence_model.encode(input_original_text)
        changed_emb = sentence_model.encode(input_changed_text)

        # Compute cosine similarity between the original and changed news
        similarity_score = cosine_similarity([original_emb], [changed_emb])[0][0]

        st.write(f"Similarity Score: {similarity_score:.2f}")

        # If the similarity score is above a threshold, consider it similar
        if similarity_score > 0.8:  # Threshold for similarity
            st.info("The news has been altered but remains similar to the original.")
        else:
            st.warning("The news is significantly different from the original.")


