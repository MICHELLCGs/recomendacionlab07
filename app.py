import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import classification_report, confusion_matrix


def manhattan(rating1, rating2):
    """Calcula la distancia de Manhattan. Ambos rating1 y rating2
    son diccionarios del tipo {'wordcount': valor, 'sentimentValue': valor...}"""
    
    distance = 0
    for key in rating1:
        if key in rating2:
            distance += abs(rating1[key] - rating2[key])
    return distance

def computeNearestNeighbor(instance, data):
    """Crea una lista ordenada de las instancias basadas en la distancia a `instance`."""    
    distances = []
    for index, row in data.iterrows():

        if not np.array_equal(instance, row):
            dist = manhattan(dict(zip(instance.index, instance.values)), 
                             dict(zip(row.index, row.values)))
            distances.append((dist, index))  
    distances.sort()  
    return distances


dataframe = pd.read_csv(r"reviews_sentiment.csv", sep=';')

print(dataframe.head(10))


X = dataframe[['wordcount', 'sentimentValue']]
y = dataframe['Star Rating']


X_train = X.sample(frac=0.8, random_state=0)
X_test = X.drop(X_train.index)
y_train = y[X_train.index]
y_test = y[X_test.index]


def predict_rating(instance, X_train, y_train, k=7):
    """Predice el 'Star Rating' para una instancia dada usando los vecinos más cercanos."""    

    neighbors = computeNearestNeighbor(instance, X_train)
    

    neighbor_labels = [y_train[idx] for _, idx in neighbors[:k]]
    

    prediction = max(set(neighbor_labels), key=neighbor_labels.count)
    
    return prediction


predictions = []
for idx, row in X_test.iterrows():
    pred = predict_rating(row, X_train, y_train)
    predictions.append(pred)


print("Matriz de Confusión:")
print(confusion_matrix(y_test, predictions))

print("\nReporte de Clasificación:")
print(classification_report(y_test, predictions))


def accuracy(y_true, y_pred):
    """Calcula la precisión manualmente."""    
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)

k_range = range(1, 20)
scores = []

for k in k_range:
    predictions_k = []
    for idx, row in X_test.iterrows():
        pred = predict_rating(row, X_train, y_train, k=k)
        predictions_k.append(pred)
    scores.append(accuracy(y_test, predictions_k))

plt.figure()
plt.plot(k_range, scores, marker='o')
plt.xlabel('k')
plt.ylabel('Precisión')
plt.title('Precisión en función de k (Distancia de Manhattan)')
plt.xticks(k_range)
plt.grid(True)
plt.show()


best_k = k_range[np.argmax(scores)]
print(f"El mejor valor de k es: {best_k} con una precisión de {max(scores):.2f}")

#Ejemplo
example_instance = pd.Series({'wordcount': 15, 'sentimentValue': -0.5})
predicted_rating = predict_rating(example_instance, X_train, y_train, k=best_k)

print(f"La calificación predicha para wordcount=15 y sentimentValue=-0.5 es: {predicted_rating}")
