# Kaggle---LLM-Science-Exam

## Introduction
This repository contains the work and achievements for the Kaggle NLP competition, "LLM-Science-Exam". Our approach ranked in the top 3.8% of participants, securing a Silver Medal with a Mean Average Precision (MAP) of 0.905022.

## Project Highlights
Competition Achievement: Attained an impressive MAP of 0.905022, ranking 103rd out of 2665 participants.

Advanced Data Augmentation & Retrieval Techniques: Implemented an innovative augmentation strategy and used GTE-base embeddings with Faiss for efficient indexing, significantly enhancing retrieval accuracy.

LLM Optimization & Domain Adaptation: Customized and fine-tuned Llama2-70B-Chat and DeBERTa-v3-large models, integrating RAG for improved question-answering performance in a science domain.

## Repository Structure
generate_faiss_index.py: Script for generating Faiss index for efficient data retrieval.

inference-platypus2-70b-with-RAG.ipynb: Jupyter notebook for inference using Platypus2-70B model with RAG.

inference_deberta_v3_large_RAG.ipynb: Inference notebook for DeBERTa-v3-large with RAG.

llm-se-data-generation-multiquestion.ipynb: Notebook for data generation with multi-question support.

train_deberta_v3_large.py: Training script for the DeBERTa-v3-large model.
