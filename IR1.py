
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required packages (only once)
nltk.download('punkt')
nltk.download('stopwords')

# Input text
text = "The cats are playing happily in the garden."

# Convert to lowercase
text = text.lower()

# Tokenize text
words = word_tokenize(text)

# Remove stopwords
filtered_words = [word for word in words if word not in stopwords.words('english') and word.isalpha()]

# Apply stemming
ps = PorterStemmer()
stemmed_words = [ps.stem(word) for word in filtered_words]

print("Original Text: ", text)
print("After Stopword Removal: ", filtered_words)
print("After Stemming: ", stemmed_words)
