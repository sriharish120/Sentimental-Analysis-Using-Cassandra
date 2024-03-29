import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
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
print(df)
#df.to_csv("C:\\Users\\SRIHARISH\\Downloads\\processednosql.csv",index=False)
X = df['word']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
predictions = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
classification_rep = classification_report(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')
