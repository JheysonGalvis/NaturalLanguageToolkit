import nltk
import random
# Ejemplo de conjunto de datos de textos etiquetas
data = [
("Me encanta esta película", "positivo"),
("Esta película es terrible", "negativo"),
("Esta película es genial", "positivo"),
("No me gusta esta película", "negativo"),
("Esta película es increíble", "positivo"),
("No soporto ver esta película", "negativo"), 
("La actuación en esta película es fenomenal", "positivo"),
("Me arrepiento de haber perdido el tiempo en esta película", "negativo"),
("Disfruté mucho esta película", "positivo"),
("A esta película le falta profundidad y sustancia", "negativo"),
("La trama de esta película fue cautivadora", "positivo"),
("Encontré los personajes de esta película muy atractivos", "positivo"),
("Los efectos especiales de esta película fueron impresionantes", "positivo"), 
("La historia era predecible y poco original", "negativo"),
("Me decepcionó la falta de desarrollo del personaje", "negativo"),
("La fotografía en esta película fue impresionante", "positivo"),
("El diálogo parecía forzado y poco natural", "negativo"),
("El ritmo de la película fue demasiado lento para mi gusto", "negativo"),
("Me sorprendió gratamente lo mucho que disfruté esta película", "positivo"),
("El final me dejó insatisfecho y confundido", "negativo"), 
("Esta película superó mis expectativas", "positivo"),
("Las actuaciones de los actores fueron mediocres", "negativo")
]
# Preprocesamiento de datos: tokenización y extracción de características
def preprocess(text):
    tokens = nltk.word_tokenize(text)
    return {words: True for words in tokens}
# Aplicamos el preprocesamiento a los datos
featuresets = [(preprocess(text), label) for (text, label) in data]
# Dividimos los datos en conjuntos de entrenamiento y prueba 
train_set, test_set = featuresets[:16], featuresets[16:]
# Entrenamos un clasificador utilizando Naive Bayes
classifier = nltk.NaiveBayesClassifier.train(train_set)
# Evaluamos el clasificador en el conjunto de prueba
accuracy = nltk.classify.accuracy(classifier, test_set)
print("Accuracy:", accuracy)
# Clasificamos un nuevo texto
new_text = "Esta película es genial"
new_text_features = preprocess(new_text)
predicted_label = classifier.classify(new_text_features)
print("Predicted label:", predicted_label)