
# Import ArXiv articles dataset

# Load data from huggingface 
from datasets import load_dataset
dataset = load_dataset("maartengr/arxiv_nlp")["train"]

# Extract metadata
abstracts = dataset["Abstracts"]
titles = dataset["Titles"]

# Common pipeline for text clustering

# 1) Embedding documetns

from sentence_transformers import SentenceTransformer

# Create an embeddings for each abstract
embedding_model = SentenceTransformer('thenlper/gte-small')
embeddings = embedding_model.encode(abstracts, show_progress_bar=True)

# Check the dimensions of the resulting embeddings
# (number_of_documents, embeddings dimensions)
print(embeddings.shape)


# 2) Reducing the Dimensionality of Embeddings

from umap import UMAP

# We reduce the input embeddings from 384 dimensions to 5 dimensions
umap_model = UMAP(
    n_components=5, min_dist=0.0, metric='cosine', random_state=42
)
reduced_embeddings = umap_model.fit_transform(embeddings)

print(f"reduced dimmensions: {reduced_embeddings.shape[0]},{reduced_embeddings.shape[1]}")

# 3 Cluster the reduced embeddings
from hdbscan import HDBSCAN

# We fit the model and extract the clusters
hdbscan_model = HDBSCAN(
    min_cluster_size=50, metric='euclidean', cluster_selection_method='eom'
).fit(reduced_embeddings)
clusters = hdbscan_model.labels_

# How many clusters did we generate?
print(len(set(clusters)))

# Inspecgting the clusters

import numpy as np

# Print first three documents in cluster 0
cluster = 0
for index in np.where(clusters==cluster)[0][:3]:
    print(abstracts[index][:300] + "... \n")

import pandas as pd

# Reduce 384-dimensional embeddings to 2 dimensions for easier visualization
reduced_embeddings = UMAP(
    n_components=2, min_dist=0.0, metric='cosine', random_state=42
).fit_transform(embeddings)

# Create dataframe
df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
df["title"] = titles
df["cluster"] = [str(c) for c in clusters]

# Select outliers and non-outliers (clusters)
clusters_df = df.loc[df.cluster != "-1", :]
outliers_df = df.loc[df.cluster == "-1", :]

import matplotlib.pyplot as plt

# Plot outliers and non-outliers seperately
plt.scatter(outliers_df.x, outliers_df.y, alpha=0.05, s=2, c="grey")
plt.scatter(
    clusters_df.x, clusters_df.y, c=clusters_df.cluster.astype(int),
    alpha=0.6, s=2, cmap='tab20b'
)
plt.axis('off')
# plt.savefig("matplotlib.png", dpi=300)  # Uncomment to save the graph as a .png

# From Text Clustering to Topic Modeling

# BERTopic: A Modular Topic Modeling Framework

from bertopic import BERTopic

# Train our model with our previously defined models
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True
).fit(abstracts, embeddings)

# Explore the topics 
print(topic_model.get_topic_info())

# Get top 10 keywords per topic
print(topic_model.get_topic(0))

# Let's search for a topic about topic modeling:
print(topic_model.find_topics("topic modeling"))

# Topic 22 has a relatively high similarity with our search term
print(topic_model.get_topic(22))

print(topic_model.topics_[titles.index('BERTopic: Neural topic modeling with a class-based TF-IDF procedure')])

# Visualizations

# Visualize topics and documents
fig = topic_model.visualize_documents(
    titles,
    reduced_embeddings=reduced_embeddings,
    width=1200,
    hide_annotations=True
)

# Update fonts of legend for easier visualization
fig.update_layout(font=dict(size=16))

# Visualize barchart with ranked keywords
topic_model.visualize_barchart()

# Visualize relationships between topics
topic_model.visualize_heatmap(n_clusters=30)

# Visualize the potential hierarchical structure of topics
topic_model.visualize_hierarchy()


# Representation models


# Save original representations
from copy import deepcopy
original_topics = deepcopy(topic_model.topic_representations_)

def topic_differences(model, original_topics, nr_topics=5):
    """Show the differences in topic representations between two models """
    df = pd.DataFrame(columns=["Topic", "Original", "Updated"])
    for topic in range(nr_topics):

        # Extract top 5 words per topic per model
        og_words = " | ".join(list(zip(*original_topics[topic]))[0][:5])
        new_words = " | ".join(list(zip(*model.get_topic(topic)))[0][:5])
        df.loc[len(df)] = [topic, og_words, new_words]

    return df

from bertopic.representation import KeyBERTInspired

# Update our topic representations using KeyBERTInspired
representation_model = KeyBERTInspired()
topic_model.update_topics(abstracts, representation_model=representation_model)

# Show topic differences
print(topic_differences(topic_model, original_topics))

# Maximal Marginal Relevance
from bertopic.representation import MaximalMarginalRelevance

# Update our topic representations to MaximalMarginalRelevance
representation_model = MaximalMarginalRelevance(diversity=0.5)
topic_model.update_topics(abstracts, representation_model=representation_model)

# Show topic differences
print(topic_differences(topic_model, original_topics))

# Text generation
# Flan - T5
from transformers import pipeline
from bertopic.representation import TextGeneration

prompt = """I have a topic that contains the following documents:
[DOCUMENTS]

The topic is described by the following keywords: '[KEYWORDS]'.

Based on the documents and keywords, what is this topic about?"""

# Update our topic representations using Flan-T5
generator = pipeline('text2text-generation', model='google/flan-t5-small')
representation_model = TextGeneration(
    generator, prompt=prompt, doc_length=50, tokenizer="whitespace"
)
topic_model.update_topics(abstracts, representation_model=representation_model)

# Show topic differences
topic_differences(topic_model, original_topics)


# OpenAI

# import openai
# from bertopic.representation import OpenAI

# prompt = """
# I have a topic that contains the following documents:
# [DOCUMENTS]

# The topic is described by the following keywords: [KEYWORDS]

# Based on the information above, extract a short topic label in the following format:
# topic: <short topic label>
# """

# # Update our topic representations using GPT-3.5
# client = openai.OpenAI(api_key="YOUR_KEY_HERE")
# representation_model = OpenAI(
#     client, model="gpt-3.5-turbo", exponential_backoff=True, chat=True, prompt=prompt
# )
# topic_model.update_topics(abstracts, representation_model=representation_model)

# # Show topic differences
# topic_differences(topic_model, original_topics)

# # Visualize topics and documents
# fig = topic_model.visualize_document_datamap(
#     titles,
#     topics=list(range(20)),
#     reduced_embeddings=reduced_embeddings,
#     width=1200,
#     label_font_size=11,
#     label_wrap_width=20,
#     use_medoids=True,
# )
# plt.savefig("datamapplot.png", dpi=300)

# Word Cloud

# topic_model.update_topics(abstracts, top_n_words=500)

# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# def create_wordcloud(model, topic):
#     plt.figure(figsize=(10,5))
#     text = {word: value for word, value in model.get_topic(topic)}
#     wc = WordCloud(background_color="white", max_words=1000, width=1600, height=800)
#     wc.generate_from_frequencies(text)
#     plt.imshow(wc, interpolation="bilinear")
#     plt.axis("off")
#     plt.show()

# # Show wordcloud
# create_wordcloud(topic_model, topic=17)