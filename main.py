import joblib
import re
from nltk.corpus import stopwords

model = joblib.load('fake_news_model.joblib')
vectorization = joblib.load('tfidf_vectorizer.joblib')


def preprocess_new_text(text_data):
    sentence = re.sub(r'[^\w\s]', '', text_data)
    preprocessed_text = ' '.join(token.lower()
                                 for token in str(sentence).split()
                                 if token.lower() not in stopwords.words('english'))
    return preprocessed_text


fake_news = {
            "gpt_generated": ["""September 11, 2023
In a shocking turn of events, newly declassified government documents reveal that the 9/11 attacks were not orchestrated by Al-Qaeda, but by high-ranking officials within the U.S. government. According to these documents, explosives were planted inside the World Trade Center (WTC) buildings weeks before the attacks. Multiple eyewitness accounts and whistleblowers have come forward, claiming they saw suspicious activities involving government personnel in the days leading up to the event.

Explosives Inside the Towers
Contrary to the official narrative that the towers collapsed due to the impact of the hijacked planes, engineers who have reviewed the documents suggest that a controlled demolition occurred. Explosives were strategically placed inside both WTC towers, ensuring a symmetrical collapse. This corroborates several conspiracy theories that have been circulating for years.

The Government's Role
Former intelligence operatives, now stepping forward, allege that the U.S. government orchestrated the attacks as a "false flag" operation to justify military action in the Middle East. The goal, they claim, was to secure oil interests and expand U.S. geopolitical dominance. The documents detail how key officials planned the event and used it to manipulate public sentiment, passing the controversial Patriot Act and ramping up military spending.

Media Cover-Up
Mainstream media outlets, the whistleblowers allege, were complicit in covering up the truth. A secret agreement between media giants and the U.S. government ensured that the official version of the events dominated airwaves, while dissenting voices were silenced. Popular news stations refused to air any alternative viewpoints or investigate the suspicious collapse of WTC Building 7, which fell without being hit by a plane.

This bombshell revelation could change everything we thought we knew about one of the darkest days in American history."""],
            "web": ["""During the visit, Zelensky gave his first in-person address to the UN General Assembly, met with lawmakers on Capitol Hill, and also visited the White house. While Zelensky was blitzing Washington in urgent effort to bolster support for Ukraine, his wife Olena Zelenska was spotted on Fifth Avenue in NYC.

New York’s Fifth Avenue is the city’s most famous shopping street, and probably the most famous shopping street in the world. A lot of prestigious and high-end stores can be found between 49th and 60th Street on Fifth Avenue including Armani, Gucci, Bergdorf Goodman, Harry Winston, Cartier. Modern day celebrities are often spotted wearing CARTIER pieces: Angelina Jolie, Kylie Jenner, Lupita Nyong’o and … Olena Zelenska. According to our sources, the wife of Ukrainian president is a diehard Cartier enthusiast. Moreover, she has even visited the famous Cartier Mansion during Ukrainian

President’s visit to NYC to address the United Nations General Assembly, and has reportedly spent $1,100,000 on jewelry.

According to information collected by Boukari Ouédraogo from the Cartier store ex-employee, Olena Zelenska visited the boutique during her and her husband’s visit to New York. “I tried to take her on a quick tour, but she wasn’t interested,” the ex-employee further recalls.
Zelenska’s visit to the luxury boutique ended up in a very unexpected manner as she snapped at the employee who was trying to assist her with a “Who said I need your opinion?” rant. After that, according to the boutique ex-worker, Zelenska had a talk with the manager. The ex-worker has no idea what the discussion was about but the next day she got fired from the boutique.
After receiving a “you’re fired” call the next day after Zelenska’s visit, the ex-employee decided to share her story about the bizarre encounter on the Instagram. She has managed to sneak away a copy of a receipt containing Zelenska’s purchases while packing her personal belongings at the boutique.
"""]}

preprocessed_text = preprocess_new_text(fake_news.get("web")[0])
transformed_text = vectorization.transform([preprocessed_text])

# Načítanie uložených modelov
lr_model = joblib.load('trained_models/fake_news/fake_news_lr_model.joblib')
svc_model = joblib.load('trained_models/fake_news/fake_news_svc_model.joblib')
rf_model = joblib.load('trained_models/fake_news/fake_news_rf_model.joblib')

# Pravdepodobnostné predikcie
lr_prob = lr_model.predict_proba(transformed_text)
svc_prob = svc_model.predict_proba(transformed_text)
rf_prob = rf_model.predict_proba(transformed_text)

print(f"Predikcia logistickej regresie: {lr_prob}")
print(f"Predikcia Support Vector Classifier (SVC): {svc_prob}")
print(f"Predikcia Random Forest modelu: {rf_prob}")

# Priemerovanie pravdepodobností
avg_prob = (lr_prob + svc_prob + rf_prob) / 3
ensemble_prediction = avg_prob.argmax(axis=1)
print(f"Konecna predikcia: {ensemble_prediction}")

print("Real News" if ensemble_prediction[0] == 1 else "Fake News")
