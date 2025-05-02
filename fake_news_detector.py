# import pandas as pd
# import string
# import nltk
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # Download NLTK stopwords
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))

# # 1. Load and label data
# fake_df = pd.read_csv(r"fakeNews\Fake.csv")
# true_df = pd.read_csv(r"fakeNews\True.csv")

# fake_df["label"] = 1  # 1 = Fake
# true_df["label"] = 0  # 0 = Real

# df = pd.concat([fake_df, true_df], ignore_index=True)
# df = df[["text", "label"]].dropna()

# # 2. Text cleaning
# def clean_text(text):
#     text = text.lower()
#     text = ''.join([ch for ch in text if ch not in string.punctuation])
#     words = text.split()
#     words = [w for w in words if w not in stop_words]
#     return ' '.join(words)

# df["cleaned_text"] = df["text"].apply(clean_text)

# # 3. TF-IDF Vectorization
# tfidf = TfidfVectorizer(max_features=5000)
# X = tfidf.fit_transform(df["cleaned_text"]).toarray()
# y = df["label"]

# # 4. Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 5. Model training
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # 6. Evaluation
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))

# # 7. Save model and vectorizer
# with open(r"fakeNews\model.pkl", "wb") as f:
#     pickle.dump(model, f)

# with open(r"fakeNews\vectorizer.pkl", "wb") as f:
#     pickle.dump(tfidf, f)




# import streamlit as st
# import pandas as pd
# import string
# import nltk
# import pickle
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

# # --- Section 1: Upload or paste custom dataset ---
# st.subheader("📂 Upload a Dataset or Paste Your Data")
# uploaded_file = st.file_uploader("Upload a CSV file (must have 'text' and 'label' columns)", type="csv")

# paste_mode = st.checkbox("Or paste data manually")

# if paste_mode:
#     text_data = st.text_area("Paste your data here (text,label):", height=150)
#     if text_data:
#         lines = text_data.strip().split('\n')
#         data = [line.split(',') for line in lines if ',' in line]
#         df = pd.DataFrame(data, columns=["text", "label"])
#         df['label'] = df['label'].astype(int)
#     else:
#         df = None
# elif uploaded_file:
#     df = pd.read_csv(uploaded_file)
# else:
#     df = None

# # --- Section 2: Train Model ---
# if df is not None:
#     st.write("📊 Preview of your data:")
#     st.dataframe(df.head())

#     if st.button("🧠 Train Model"):
#         try:
#             df["cleaned_text"] = df["text"].apply(clean_text)
#             tfidf = TfidfVectorizer(max_features=5000)
#             X = tfidf.fit_transform(df["cleaned_text"]).toarray()
#             y = df["label"]

#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#             model = LogisticRegression()
#             model.fit(X_train, y_train)

#             acc = accuracy_score(y_test, model.predict(X_test))
#             st.success(f"✅ Model trained successfully! Accuracy: {acc:.2f}")

#             # Save model and vectorizer to session state
#             st.session_state['model'] = model
#             st.session_state['vectorizer'] = tfidf
#         except Exception as e:
#             st.error(f"❌ Error in training: {e}")

# # --- Section 3: Check a news article ---
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
#         st.warning("⚠️ Please upload and train a model first.")



# import pandas as pd
# import string
# import nltk
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # Download NLTK stopwords
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))

# # 1. Load and label data
# fake_df = pd.read_csv(r"fakeNews\Fake.csv")
# true_df = pd.read_csv(r"fakeNews\True.csv")

# fake_df["label"] = 1  # 1 = Fake
# true_df["label"] = 0  # 0 = Real

# df = pd.concat([fake_df, true_df], ignore_index=True)
# df = df[["text", "label"]].dropna()

# # 2. Text cleaning function
# def clean_text(text):
#     text = text.lower()
#     text = ''.join([ch for ch in text if ch not in string.punctuation])
#     words = text.split()
#     words = [w for w in words if w not in stop_words]
#     return ' '.join(words)

# df["cleaned_text"] = df["text"].apply(clean_text)

# # 3. TF-IDF Vectorization
# tfidf = TfidfVectorizer(max_features=5000)
# X = tfidf.fit_transform(df["cleaned_text"]).toarray()
# y = df["label"]

# # 4. Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 5. Model training
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # 6. Evaluation
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))

# # 7. Save model and vectorizer
# with open("model.pkl", "wb") as f:
#     pickle.dump(model, f)

# with open("vectorizer.pkl", "wb") as f:
#     pickle.dump(tfidf, f)




# import pandas as pd
# import string
# import nltk
# import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report

# # Download NLTK stopwords
# nltk.download('stopwords')
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))

# # 1. Load and label data
# fake_df = pd.read_csv(r"fakeNews\Fake.csv")
# true_df = pd.read_csv(r"fakeNews\True.csv")

# fake_df["label"] = 1  # 1 = Fake
# true_df["label"] = 0  # 0 = Real

# df = pd.concat([fake_df, true_df], ignore_index=True)
# df = df[["text", "label"]].dropna()

# # 2. Text cleaning function
# def clean_text(text):
#     text = text.lower()
#     text = ''.join([ch for ch in text if ch not in string.punctuation])
#     words = text.split()
#     words = [w for w in words if w not in stop_words]
#     return ' '.join(words)

# df["cleaned_text"] = df["text"].apply(clean_text)

# # 3. TF-IDF Vectorization
# tfidf = TfidfVectorizer(max_features=5000)
# X = tfidf.fit_transform(df["cleaned_text"]).toarray()
# y = df["label"]

# # 4. Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 5. Model training
# model = LogisticRegression()
# model.fit(X_train, y_train)

# # 6. Evaluation
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Classification Report:\n", classification_report(y_test, y_pred))

# # 7. Save model and vectorizer
# with open("model.pkl", "wb") as f:
#     pickle.dump(model, f)

# with open("vectorizer.pkl", "wb") as f:
#     pickle.dump(tfidf, f)




# import difflib
# import streamlit as st
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# # Load Sentence Transformer model for similarity comparison
# sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

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
# st.subheader("Compare the original and modified news, and detect its authenticity.")

# # Step 1: Compare the original and changed news
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




import difflib
import streamlit as st
import pickle
import string
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

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

# Function to compare and highlight changed words using difflib
def highlight_changes(original_text, changed_text):
    # SequenceMatcher gives us a ratio of similarity and identifies changed parts
    matcher = difflib.SequenceMatcher(None, original_text, changed_text)
    highlighted_text = ""

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        original_substring = original_text[i1:i2]
        changed_substring = changed_text[j1:j2]

        if tag == 'replace':  # When words are replaced, highlight both removed and added text
            highlighted_text += f"<span style='color:red; text-decoration:line-through'>{original_substring}</span> "
            highlighted_text += f"<span style='color:green'>{changed_substring}</span> "
        elif tag == 'delete':  # Text removed
            highlighted_text += f"<span style='color:red; text-decoration:line-through'>{original_substring}</span> "
        elif tag == 'insert':  # Text inserted
            highlighted_text += f"<span style='color:green'>{changed_substring}</span> "
        else:  # Text that is unchanged
            highlighted_text += original_substring

    return highlighted_text

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
        # Highlight the changes (words that are different)
        highlighted_changes = highlight_changes(input_original_text, input_changed_text)
        
        # Display the changes in Streamlit
        st.markdown("**Highlighted Changes Between Original and Changed News:**", unsafe_allow_html=True)
        st.markdown(f"<div>{highlighted_changes}</div>", unsafe_allow_html=True)

        # Encode the texts using SentenceTransformer
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
