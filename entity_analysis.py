"""
Module 6 Week A — Integration: Entity Analysis Pipeline """


import unicodedata

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch 
import spacy
from itertools import combinations


def load_corpus(filepath="data/climate_articles.csv"):
    """Load the climate articles dataset."""
    df = pd.read_csv(filepath)
    return df 

def preprocess_corpus(df):
    """Add a language-aware `processed_text` column to the corpus.

    For every row, apply Unicode NFC normalization to `text` so that
    visually identical characters (composed vs. decomposed diacritics)
    compare equal downstream. The processed form preserves
    capitalization and punctuation — those are signals NER depends on.

    For Arabic rows (`language == 'ar'`), do not attempt English NLP
    processing: either pass the NFC-normalized text through unchanged
    or store an empty string. Either choice must not crash the
    pipeline.

    Args:
        df: DataFrame returned by load_corpus.

    Returns:
        Copy of df with a new `processed_text` column. The original
        `text` column is left intact so NER can still consume it.
    """
    df_copy = df.copy()
    processed_text = []
    for _, row in df_copy.iterrows():
        text = row["text"]
        language = row["language"]

        normalized_text = unicodedata.normalize("NFC", str(text))

        if row["language"] == "en":
            processed_text.append(normalized_text)
        elif row["language"] == "ar":
            processed_text.append(normalized_text)
        else:
            processed_text.append("")
    df_copy["processed_text"] = processed_text
    return df_copy


def run_ner_pipeline(df, nlp):
    """Run spaCy NER on the English rows of a preprocessed corpus.

    Args:
        df: DataFrame with columns id, text, language, processed_text.
        nlp: A loaded spaCy Language object (e.g., en_core_web_sm).

    Returns:
        DataFrame with columns: text_id, entity_text, entity_label,
        start_char, end_char.
    """
    df_en =df[df["language"]=="en"]
    records = []
    for _, row in df_en.iterrows():
        text = row["text"]
        text_id = row["id"]
        doc = nlp(text)
        for ent in doc.ents:
            records.append({
                "text_id":row["id"],
                "entity_text":ent.text,
                "entity_label":ent.label_,
                "start_char":ent.start_char,
                "end_char":ent.end_char
        })
    entities_df = pd.DataFrame(records) 
     
    return entities_df  

def aggregate_entity_stats(entity_df, articles_df):
    """Compute frequency, co-occurrence, and per-category statistics.

    Args:
        entity_df: DataFrame with columns text_id, entity_text,
                   entity_label.
        articles_df: The source corpus DataFrame (with columns id,
                     category, ...). Used to join category onto
                     each entity for per-category aggregation.

    Returns:
        Dictionary with keys:
          'top_entities': DataFrame of top 20 entities by frequency
                          (columns: entity_text, entity_label, count)
          'label_counts': dict of entity_label -> total count
          'co_occurrence': DataFrame of entity pairs appearing in the
                           same text (columns: entity_a, entity_b,
                           co_count). Cap at top 50 pairs by co_count
                           (or filter to co_count >= 2) so the result
                           stays readable on the full corpus.
          'per_category': DataFrame of entity-label counts broken out
                          by article category (columns: category,
                          entity_label, count)
    """
    top_entities = (
        entity_df . groupby(["entity_text","entity_label"]).size().reset_index(name="count").sort_values("count",ascending = False).head(20)
    )
    label_counts = entity_df["entity_label"].value_counts().to_dict()
    pairs = []
    for text_id,group in entity_df.groupby("text_id"):
        unique_entities = group["entity_text"].unique()
        for a,b in combinations(sorted(unique_entities),2):
            pairs.append((a,b))
    co_df = pd.DataFrame(pairs,columns=["entity_a","entity_b"])
    co_occurrence =(

    co_df.value_counts().reset_index(name = "co_count").sort_values("co_count", ascending =False)
      )        
    co_occurrence = co_occurrence[co_occurrence["co_count"] >=2].head(50)

    merged = entity_df.merge( articles_df,left_on ="text_id",right_on ="id")

    per_category = (merged.groupby(["category","entity_label"]).size().reset_index(name="count"))
    print("Top entities :\n",top_entities.head())
    print ("\n Label counts:\n",label_counts)
    print ("\n Top co-occurrence:\n",co_occurrence.head())
    print("\n per category:\n",per_category.head())

    return {
        "top_entities": top_entities,
        "label_counts":label_counts,
        "co_occurrence":co_occurrence,
        "per_category":per_category
    }

def visualize_entity_distribution(stats, output_path="entity_distribution.png"):
    """Create a bar chart of the top 20 entities by frequency.

    Args:
        stats: Dictionary from aggregate_entity_stats (must contain
               'top_entities' DataFrame).
        output_path: File path to save the chart.
    """
    top_entities = stats["top_entities"].sort_values("count", ascending = True)
    labels = top_entities["entity_label"].unique()
    colors= plt.cm.tab10(range(len(labels)))
    color_map = dict(zip(labels,colors))
    bar_colors = top_entities["entity_label"].map(color_map)

    plt.figure(figsize=(10,8))
    plt.barh(
        top_entities["entity_text"],
        top_entities["count"],
        color = bar_colors
    )
    plt.xlabel("Frequency")
    plt.ylabel("Entity")
    plt.title("Top 20 Most Frequent Entities")

    legend_elements = [Patch(facecolor = color_map[label], label=label) 
                       for label in color_map]
    plt.legend(handles = legend_elements, title = "Entity Type")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_report(stats, co_occurrence):
    """Generate a text summary of entity analysis findings.

    Args:
        stats: Dictionary from aggregate_entity_stats.
        co_occurrence: Co-occurrence DataFrame from stats.

    Returns:
        String containing a structured report with: entity counts
        per type, top 5 most frequent entities, top 3 co-occurring
        pairs, and a brief summary.
    """
    label_counts = stats["label_counts"]
    top5 = stats["top_entities"].head(5)
    top3_pairs = co_occurrence.head(3)

    counts_str="\n".join(
        [f"-{label}:{count}" for label, count in label_counts.items()]
    )
    top_entities_str = "\n".join([
        f"-{row.entity_text}({row.entity_label}):{row['count']}" 
        for _,row in top5.iterrows()]
    )
    top3_pairs_str = "\n".join([
        f"-{row.entity_a} - {row.entity_b} ({row['co_count']})"
        for _,row in top3_pairs.iterrows()]
    )
    summary = (
        "The dataset is heavily dominated by DATA entities (256), driven by frequent mentions of specific organizations, locations,"
        " and actors relevant to climate discussions for years like 2030 and 2023. "
        "This suggests a strong emphasis on timelines and future climate targets "
        "Geographic entities such as Jordan(GPE) aslo appear frequently, "
        " indicate regional focus .Organizations (ORG) are also prominent, reflecting the "
        " consistently mentioned together, reflecting between institutions, regions,"
        "institutional nature of climate discussions."

    )
    report = f""" 
    ===Entity Report===
    Entity Counts by Type:
    {counts_str}

    Top 5 Entities:
    {top_entities_str}

    Top 3 Co-occurring Entity Pairs:
    {top3_pairs_str}
    
    Summary:
    {summary}"""
    return report.strip()

if __name__ == "__main__":
    nlp = spacy.load("en_core_web_sm")

    # Load and preprocess the corpus
    raw = load_corpus()
    if raw is not None:
        corpus = preprocess_corpus(raw)
        if corpus is not None:
            print(f"Corpus: {len(corpus)} articles")
            print(f"Languages: {corpus['language'].value_counts().to_dict()}")
            print(f"Categories: {corpus['category'].value_counts().to_dict()}")

            # Run NER on English rows
            entities = run_ner_pipeline(corpus, nlp)
            if entities is not None:
                print(f"\nExtracted {len(entities)} entities")

                # Aggregate statistics
                stats = aggregate_entity_stats(entities, corpus)
                if stats is not None:
                    print(f"\nLabel counts: {stats['label_counts']}")
                    print(f"\nTop 5 entities:")
                    print(stats["top_entities"].head())
                    print(f"\nPer-category counts (head):")
                    print(stats["per_category"].head())

                    # Visualize
                    visualize_entity_distribution(stats)
                    print("\nVisualization saved to entity_distribution.png")

                    # Generate report
                    report = generate_report(stats, stats.get("co_occurrence"))
                    if report is not None:
                        print(f"\n{'='*50}")
                        print(report)
