import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay



# algorithms (Models)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

import joblib # package to build model


data = pd.read_csv('dataset/news_data.csv', index_col=0)
data = data.drop(["title", "subject", "date"], axis=1)

data = data.sample(frac=1)
data.reset_index(inplace=True)
data.drop(["index"], axis=1, inplace=True)

sns.countplot(data=data, x='class', order=data['class'].value_counts().index)
plt.show()

def preprocess_text(text_data):
    """
        Removing Punctuation: Uses regex to remove punctuation from each sentence.
        Tokenization and Stopword Removal: Splits sentences into words, converts them to lowercase, and removes common English stopwords.
        Updating Data: Replaces the original text with the preprocessed text.
    """
    preprocessed_text = []

    for sentence in tqdm(text_data):
        sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove punctuation
        preprocessed_text.append(' '.join(token.lower()
                                          for token in str(sentence).split()
                                          if token.lower() not in stopwords.words('english')))
    return preprocessed_text

preprocessed_review = preprocess_text(data['text'].values)
data['text'] = preprocessed_review

# Train-Test Split: Divides the dataset into training and testing sets with a 75-25 split.
x_train, x_test, y_train, y_test = train_test_split(data['text'], data['class'], test_size=0.25)


# TF-IDF Vectorizer: Converts text data into numerical features based on Term Frequency-Inverse Document Frequency.
# Fitting and Transforming: Learns vocabulary from training data and transforms both training and testing data into TF-IDF features.
vectorization = TfidfVectorizer()
x_train = vectorization.fit_transform(x_train)
x_test = vectorization.transform(x_test)

# Model Training: Fits the logistic regression model to the training data.
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)


# Training SVC model
svc_model = SVC(kernel='linear', probability=True)
svc_model.fit(x_train, y_train)

# Training Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)


# Predikcia a vyhodnotenie logistickej regresie
lr_train_pred = lr_model.predict(x_train)
lr_test_pred = lr_model.predict(x_test)

print("Logistic Regression Training Accuracy:", accuracy_score(y_train, lr_train_pred))
print("Logistic Regression Testing Accuracy:", accuracy_score(y_test, lr_test_pred))

cm_lr = confusion_matrix(y_test, lr_test_pred)
cm_display_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=[False, True])
cm_display_lr.plot()
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# Predikcia a vyhodnotenie SVC modelu
svc_train_pred = svc_model.predict(x_train)
svc_test_pred = svc_model.predict(x_test)

print("SVC Training Accuracy:", accuracy_score(y_train, svc_train_pred))
print("SVC Testing Accuracy:", accuracy_score(y_test, svc_test_pred))

cm_svc = confusion_matrix(y_test, svc_test_pred)
cm_display_svc = ConfusionMatrixDisplay(confusion_matrix=cm_svc, display_labels=[False, True])
cm_display_svc.plot()
plt.title('Confusion Matrix - SVC')
plt.show()

# Predikcia a vyhodnotenie Random Forest modelu
rf_train_pred = rf_model.predict(x_train)
rf_test_pred = rf_model.predict(x_test)

print("Random Forest Training Accuracy:", accuracy_score(y_train, rf_train_pred))
print("Random Forest Testing Accuracy:", accuracy_score(y_test, rf_test_pred))

cm_rf = confusion_matrix(y_test, rf_test_pred)
cm_display_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf, display_labels=[False, True])
cm_display_rf.plot()
plt.title('Confusion Matrix - Random Forest')
plt.show()


# Pravdepodobnostné predikcie
lr_probs = lr_model.predict_proba(x_test)
svc_probs = svc_model.predict_proba(x_test)
rf_probs = rf_model.predict_proba(x_test)

# Priemerovanie pravdepodobností
avg_probs = (lr_probs + svc_probs + rf_probs) / 3
# Konečná predikcia
ensemble_pred = avg_probs.argmax(axis=1)

print("Ensemble Testing Accuracy:", accuracy_score(y_test, ensemble_pred))

# Konfúzna matica pre ensemblovaný model
cm_ensemble = confusion_matrix(y_test, ensemble_pred)
cm_display_ensemble = ConfusionMatrixDisplay(confusion_matrix=cm_ensemble, display_labels=[False, True])
cm_display_ensemble.plot()
plt.title('Confusion Matrix - Ensemble Model')
plt.show()


# Uloženie modelov a vektorizéra
joblib.dump(lr_model, 'trained_models/fake_news/fake_news_lr_model.joblib')
joblib.dump(svc_model, 'trained_models/fake_news/fake_news_svc_model.joblib')
joblib.dump(rf_model, 'trained_models/fake_news/fake_news_rf_model.joblib')
joblib.dump(vectorization, 'trained_models/fake_news/tfidf_vectorizer.joblib')
