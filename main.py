import nltk 
#from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

words = ["running", "plays", "jumped"] 
stemmer = PorterStemmer()
stems = [stemmer.stem(word) for word in words] 
print(stems)

#nltk.download('punkt') 

#sentence = "NLTK es una biblioteca de procesamiento de lenguaje natural"
#tokens = word_tokenize(sentence)
#print(tokens)

