import spacy
import requests
from bs4 import BeautifulSoup
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")

# Clean up the text by removing stopwords
def remove_stopwords(text):
    tokens = text.split()
    return ' '.join([word for word in tokens if word.lower() not in stopwords.words('english')])

# Extract entities subject, predicate, and object from the text
def extract_entities(text):
    doc = nlp(text)
    subject = None
    predicate = None
    obj = None

    for token in doc:
        # Identify the subject, predicate, and object
        if "subj" in token.dep_:
            subject = remove_stopwords(token.text)
        elif token.dep_ in ["attr", "dobj", "obj"]:
            obj = remove_stopwords(token.text)
        elif token.dep_ == "ROOT":
            predicate = token.text

    # If no subject or object, go to named entities
    if not subject or not obj:
        for ent in doc.ents:
            if not subject and ent.label in ["PERSON", "ORG", "GPE", "NOUN"]:
                subject = remove_stopwords(ent.text)
            if not obj and ent.label in ["NOUN", "PRODUCT", "SUBSTANCE"]:
                obj = remove_stopwords(ent.text)

    # If still no subject or object, go to noun chunks
    if not subject or not obj:
        for chunk in doc.noun_chunks:
            if not subject:
                subject = remove_stopwords(chunk.text)
            if not obj:
                obj = remove_stopwords(chunk.text)

    # Print extracted entities for debugging if we can 
    print(f"Extracted - Subject: {subject}, Predicate: {predicate}, Object: {obj}")
    return subject, predicate, obj

def get_relation(predicate):

    # Define the mapping of predicates to relations for ConceptNet
    relation_map = {
    "is": "IsA",                
    "used": "UsedFor",
    "capable": "CapableOf",
    "part": "PartOf",
    "made": "MadeOf",
    "has": "HasA",              
    "contains": "Contains",
    "defined": "DefinedAs",
    "causes": "Causes",
    "located": "LocatedAt",
    "symbol": "SymbolOf",
    "derived": "DerivedFrom",
    "related": "RelatedTo",
    "synonym": "Synonym",
    "antonym": "Antonym",
    "motivated": "MotivatedByGoal", 
    "desires": "Desires",
    "causesdesire": "CausesDesire", 
    "distinct": "DistinctFrom", 
    "entails": "Entails",
    "atlocation": "AtLocation",
    "created": "CreatedBy",
    "instance": "InstanceOf",
    "similar": "SimilarTo",
    "receives": "ReceivesAction",
    "usedfor": "UsedFor",
    "capableof": "CapableOf",
    }

    # Find closest relation
    for key in relation_map:
        if key in predicate.lower():
            return relation_map[key]

    return None

def check_truthfulness(statement):
    # Extract subject, predicate, and object from the statement
    subject, predicate, obj = extract_entities(statement)

    # See if we can determine the subject-object relationship
    if not subject or not obj:
        return "Could not determine subject or object from statement: " + statement

    # Get the relation and check if we can use it
    relation = get_relation(predicate)
    if not relation:
        return "Could not determine relation from predicate: " + predicate
    
    # Clean up the subject and object for the API call
    subject = subject.lower().replace(" ", "_")
    obj = obj.lower().replace(" ", "_")

    # Query ConceptNet API to check the relationship
    url = f"http://api.conceptnet.io/query?start=/c/en/{subject}&rel=/r/{relation}&end=/c/en/{obj}&rel=/r/{relation}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        # print(f'Number of Edges: {data["edges"]}')  # Print the response for debugging
        # If there are relationships that are found, return True
        if data["edges"]:
            return True
        else:
            return False
        
    return "Error in API request to ConceptNet"


def main():
    # Test the check_truthfulness function
    statements = [
    "A dog is not an animal.",
    "Water is a liquid.",
    "A car has wheels.",
    "A painting is created by an artist.",
    "A cat is capable of flying.",
    "Hot weather causes desire for cold drinks."
    ]

    for statement in statements:
        result = check_truthfulness(statement)
        print(f"Statement: '{statement}' - Truthfulness: {result}")

if __name__ == "__main__":
    main()