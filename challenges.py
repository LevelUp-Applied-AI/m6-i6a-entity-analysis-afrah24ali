import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import re 
import unicodedata
from  collections import defaultdict
import spacy
import itertools
import networkx as nx

df = pd.read_csv("data/climate_articles.csv")

def extract_year(text):
    match = re.findall(r"\b(20\d{2})\b",str(text))
    return match[0] if match else None

df["year"]= df["text"].apply(extract_year)
df=df.dropna(subset=["year"])
year_counts = df.groupby("year").size()
plt.plot(year_counts.index, year_counts.values)
plt.title("Entity Trends Over Time")
plt.xlabel("year")
plt.ylabel("Mentions")
plt.show()


nlp = spacy.load("en_core_web_sm")

def extract_relationships(text):
    doc = nlp(text)
    doc_edges= []
    sent_edges = []
    
    all_entities= [ent.text for ent in doc.ents]
    for a,b in itertools.combinations(set(all_entities),2):
        doc_edges.append((a,b))
    for sent in doc.sents:
        sent_ents = [ent.text for ent in sent.ents]
        for a,b in itertools.combinations(set(sent_ents),2):
            sent_edges.append((a,b))
    return doc_edges, sent_edges
    
edges_weights = defaultdict(float)
for _,row in df.iterrows():
    doc_edges,sent_edges = extract_relationships(row["text"])
    for e in doc_edges:
        edges_weights[(e[0],e[1],"document")]+=1
    for e in sent_edges:
        edges_weights[(e[0],e[1],"sentence")]+=2

edge_df = pd.DataFrame([

    (a,b,w,level)

    for (a,b,level), w in edges_weights.items()
    ],
    columns=["entity_a","entity_b","weight","level"])   

edge_df.to_csv("data/relationship_edges.csv", index=False)

G = nx.Graph()
top_edges = edge_df.sort_values("weight", ascending=False).head(30)
for _, row in top_edges.iterrows():
    G.add_edge(row["entity_a"], row["entity_b"], weight=row["weight"], level=row["level"])
    
plt.figure(figsize=(12,8))
pos = nx.spring_layout(G, k=0.5,seed=42)
weights = [G[u][v]['weight'] for u,v in G.edges()]

nx.draw(G, pos, with_labels=True,width=[w*0.3 for w in weights], node_size=800, edge_color="gray",font_size=8)

plt.title(" Top Entity Relationship Graph")
plt.show()

nlp_base = spacy.load("en_core_web_sm")

nlp_aug = spacy.load("en_core_web_sm")
ruler = nlp_aug.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
patterns = [
    {"label":"CLIMATE_EVENT","pattern":[{"text":"COP28"}]},
    {"label": "CLIMATE_EVENT", "pattern": "COP27"},
    {"label": "POLICY", "pattern": "Paris Agreement"},
    {"label": "REPORT", "pattern": "IPCC AR6"},
    {"label": "REPORT", "pattern": "IPCC AR5"},
    {"label": "REPORT", "pattern": "Climate Change Report"},
    {"label": "THRESHOLD", "pattern": "2-degree target"},
    {"label": "THRESHOLD", "pattern": "1.5 degrees"},
    {"label": "ORGANIZATION", "pattern": "IPCC"},
    {"label": "ORGANIZATION", "pattern": "UNFCCC"},
    {"label": "ORGANIZATION", "pattern": "Climate Policy Initiative"},
    {"label": "CLIMATE_EVENT", "pattern": "Climate Summit"},
    {"label": "POLICY", "pattern": "Net Zero"},
    {"label": "POLICY", "pattern": "carbon neutrality"},
    {"label": "REPORT", "pattern": "Synthesis Report"}
]

ruler.add_patterns(patterns)

def extract_entities(nlp, text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


base_entities = []
aug_entities = []

for text in df["text"]:
    base_entities.extend(extract_entities(nlp_base, text))
    aug_entities.extend(extract_entities(nlp_aug, text))

print("BASE:", len(base_entities))
print("AUGMENTED:", len(aug_entities))
doc = nlp_aug("COP28 and Paris Agreement and IPCC AR6")
for ent in doc.ents:
    print(ent.text, ent.label_)