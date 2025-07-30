from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

example_string = """
My name is shantanu bhavsar
i like to  learn music.
It's my hobby to play guitar.
I also enjoy reading books and writing poetry.  
and I love to travel."""

# Sentence tokenization
print("\nSentence tokenization:")
print(sent_tokenize(example_string))

# Word tokenization
print("\nWord tokenization:")
words = word_tokenize(example_string)
print(words)

# Stopword removal
stop_words = set(stopwords.words('english'))
filtered_words = [word.lower() for word in words if word.lower() not in stop_words and word.isalnum()]
print("\nAfter removing stopwords:")
print(filtered_words)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
print("\nAfter lemmatization:")
print(lemmatized_words)    



