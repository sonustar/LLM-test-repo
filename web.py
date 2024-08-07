import pandas as pd
from transformers import GPT2Tokenizer, GPT2Model, Trainer, TrainingArguments
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from utils import load_yaml_file
from transformers import DataCollatorWithPadding
from torch.utils.data import Dataset
import random
from transformers import T5Tokenizer, T5ForConditionalGeneration
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re
import json

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize NLTK tools
# stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def tokenize_and_lemmatize(doc):
    """
    Tokenize, lemmatize, stem, convert to lowercase, and remove stop words from the input document.
    """

    # Extract text from HTML
    soup = BeautifulSoup(doc, "html.parser")
    # text = soup.get_text()
    
    # Clean the extracted text, including removing script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()
    
    text = soup.get_text()

    # Remove non-alphabetic characters and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    
    # Tokenize the text into individual words
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize the tokens
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return lemmatized_tokens

# Load configuration and documents
config_data = load_yaml_file("config.yaml")
loader_web = RecursiveUrlLoader(url=config_data["url"])
loader_rtdocs = ReadTheDocsLoader(config_data["folder_path"], encoding="utf-8")
loader = MergedDataLoader(loaders=[loader_web, loader_rtdocs])
docs = loader.load()

# Preprocess text
preprocessed_text = [tokenize_and_lemmatize(str(doc)) for doc in docs]
# print("PREPROCESSED TEXT")
# print(len(preprocessed_text))
# print(preprocessed_text)

# Generate synthetic data
def generate_new_text(preprocessed_text):
    """ Generate new text using a T5 text-to-text transformer model. """
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    new_text = []
    for doc in preprocessed_text:
        # Join the list of words into a single string
        doc_str = ' '.join(doc)
        # Encode the input text
        input_ids = tokenizer.encode('summarize: ' + doc_str, return_tensors='pt', truncation=True, max_length=512)
        # Generate new text using the T5 model
        output = model.generate(input_ids, max_length=100)
        # Decode the generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        new_text.append(generated_text)
    return new_text

def combine_generated_text(generated_text):
    """ Combine the generated text into synthetic documents using a simple concatenation strategy. """
    synthetic_docs = []
    for i in range(0, len(generated_text), 5):
        doc = ' '.join(generated_text[i:i+5])
        synthetic_docs.append(doc)
    return synthetic_docs

#Saves the generated-text to json in a format list of dictionary with input as key and value is the rest !! 
def save_to_json(generated_text, filename='rtdocs.json'):
    """Save paragraphs to a JSON file."""
    data = [{'input': paragraph} for paragraph in generated_text]
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Generate and combine synthetic documents
generated_text = generate_new_text(preprocessed_text)
synthetic_docs = combine_generated_text(generated_text)


print("NEW RESULTS")
print(len(generated_text))
print(generated_text)
print(synthetic_docs)

save_to_json(generated_text)
