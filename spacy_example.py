import spacy

# Load English language model
nlp = spacy.load('en_core_web_sm')


example_string = """
My name is shantanu bhavsar
i like to  learn music.
It's my hobby to play guitar.
I also enjoy reading books and writing poetry.  
and I love to travel."""

# Process the text
doc = nlp(example_string)

# Sentence tokenization
print("\nSentence tokenization:")
for sent in doc.sents:
    print(sent.text.strip())

# Word tokenization
print("\nWord tokenization:")
tokens = [token.text for token in doc]
print(tokens)

# Remove stopwords and punctuation
print("\nAfter removing stopwords and punctuation:")
filtered_tokens = [token.text for token in doc if not token.is_stop and not token.is_punct and token.text.strip()]
print(filtered_tokens)

# Lemmatization
print("\nAfter lemmatization:")
lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and token.text.strip()]
print(lemmatized_tokens)

# Additional spaCy features
print("\nPart-of-speech tagging:")
pos_tags = [(token.text, token.pos_) for token in doc if not token.is_space]
print(pos_tags)

print("\nNamed Entity Recognition:")
entities = [(ent.text, ent.label_) for ent in doc.ents]
print(entities)
