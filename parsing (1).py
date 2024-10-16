import os
import json
from transformers import pipeline
from whoosh.index import open_dir
from whoosh.qparser import QueryParser
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define your NER pipeline (BioBERT or any other medical model)
ner_pipeline = pipeline("ner", model="dmis-lab/biobert-v1.1", tokenizer="dmis-lab/biobert-v1.1", aggregation_strategy="simple")

# Load summarization model
summarizer = pipeline("summarization")

# Whoosh index directory path
index_dir = "C:\\Users\\ananya\\Desktop\\doctor\\ouput\\index"

# Function to query the Whoosh index and get file paths
def query_whoosh_index(query_text):
    ix = open_dir(index_dir)
    parser = QueryParser("content", ix.schema)
    
    with ix.searcher() as searcher:
        query = parser.parse(query_text)
        results = searcher.search(query, limit=None)  # Set limit to None to get all results
        
        # Extract the file paths from the results
        file_paths = [hit['path'] for hit in results]
        return file_paths

# Function to extract disease info from a file
def extract_disease_info(file_path, query_text):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
    except UnicodeDecodeError:
        print(f"Unicode error in file: {file_path}. Trying 'replace' mode.")
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            text = file.read()

    # Process the text with spaCy
    doc = nlp(text)

    # Initialize dictionary to store extracted information
    disease_info = {
        'symptoms': [],
        'cause': [],
        'class': []
    }

    # Define keywords for each category
    symptom_keywords = ["symptom", "sign", "indication"]
    cause_keywords = ["cause", "etiology", "origin", "source"]
    class_keywords = ["class", "type", "category", "classification"]

    # Extract information using spaCy and keywords
    for sent in doc.sents:
        sent_lower = sent.text.lower()
        if any(keyword in sent_lower for keyword in symptom_keywords):
            disease_info['symptoms'].append(sent.text.strip())
        elif any(keyword in sent_lower for keyword in cause_keywords):
            disease_info['cause'].append(sent.text.strip())
        elif any(keyword in sent_lower for keyword in class_keywords):
            disease_info['class'].append(sent.text.strip())

    # Use NER to extract additional medical entities
    ner_results = ner_pipeline(text)
    
    for entity in ner_results:
        if entity['entity_group'] == 'SYMPTOM':
            disease_info['symptoms'].append(entity['word'])
        elif entity['entity_group'] == 'CAUSE':
            disease_info['cause'].append(entity['word'])
        elif entity['entity_group'] == 'CLASS':
            disease_info['class'].append(entity['word'])

    # Filter results to include only those relevant to the query
    relevant_disease_info = {key: [] for key in disease_info}
    
    for key in disease_info:
        relevant_disease_info[key] = [item for item in disease_info[key] if query_text.lower() in item.lower()]

    return relevant_disease_info

# Summarization function
def summarize_category(category_items, max_summary_length=100):
    if not category_items:
        return ""
    
    text_to_summarize = " ".join(category_items)

    # Check if the text to summarize is too long
    if len(text_to_summarize) > 2048:  # Adjust this limit as necessary
        text_to_summarize = text_to_summarize[:2048]

    # Apply the summarization model
    summarized = summarizer(text_to_summarize, max_length=max_summary_length, min_length=30, do_sample=False)
    
    return summarized[0]['summary_text']

# Function to process multiple files, summarize results, and save to a JSON file
def process_files_and_save_summarized_results(file_paths, output_json, query_text, max_summary_length=100):
    summarized_results = {
        'symptoms': set(),
        'cause': set(),
        'class': set()
    }
    
    for file_path in file_paths:
        # Extract disease information from each file, filtering by query text
        disease_info = extract_disease_info(file_path, query_text)
        
        # Aggregate the results by adding them to the summarized results
        for key in summarized_results:
            summarized_results[key].update(disease_info[key])

    # Convert sets back to lists for summarization
    for key in summarized_results:
        summarized_results[key] = list(summarized_results[key])

    # Apply summarization for each category
    for key in summarized_results:
        summarized_results[key] = summarize_category(summarized_results[key], max_summary_length)

    # Save the summarized results to a JSON file
    with open(output_json, 'w', encoding='utf-8') as json_file:
        json.dump(summarized_results, json_file, indent=4)

# Example usage:
if __name__ == "__main__":
    # Query Whoosh index for a disease, e.g., "asthma"
    query_text = "Conjunctivitis"  # Modify this for other diseases
    file_paths = query_whoosh_index(query_text)
    print("Retrieved file paths:", file_paths)
    
    # Process the files and save the summarized results to a JSON file
    output_json = "summarized_disease_info5.json"
    process_files_and_save_summarized_results(file_paths, output_json, query_text)

    print(f"Summarized results saved to {output_json}")
