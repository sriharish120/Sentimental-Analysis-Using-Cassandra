import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from cassandra.cluster import Cluster
cluster=Cluster(["127.0.0.1"],port=9042)
session=cluster.connect("nosqlproject",wait_for_all_pools=True)
statement=f"""select * from sap;"""
rows=session.execute(statement)
instances=[]
for row in rows:
    sentiment=row.sentiment
    word=row.word
    instances.append([sentiment,word])
df=pd.DataFrame(instances,columns=["sentiment","word"])
#print(df)
X = df['word']
y = df['sentiment']
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_tfidf = tfidf_vectorizer.fit_transform(X)
y = y.map({'Positive': 1, 'Negative': 0})
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))
