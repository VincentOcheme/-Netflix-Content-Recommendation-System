# Netflix Content Recommendation System

A content-based recommendation engine that suggests similar Netflix titles using genre, director, and content-type metadata. Built as part of a machine learning internship project at Auspify Technology.

## Overview

Given any title in the dataset, this system returns the most similar titles based purely on content attributes, with no user ratings or viewing history involved. It works by converting each title's genre, director, and type into a vector, then measuring the cosine similarity between all titles in the catalog.

## How it works

1. **Feature preparation** — genre tags (`listed_in`), director, and content type are combined into a single text profile per title.
2. **Vectorization** — the combined text is converted into numeric vectors using `CountVectorizer`.
3. **Similarity scoring** — pairwise cosine similarity is computed across the full catalog.
4. **Recommendation generation** — for a given title, the top N most similar titles are returned, sorted by similarity score.
5. **Evaluation** — recommendation quality is checked using genre-overlap scoring and manual spot-checks across multiple genre categories.

## Project structure

```
netflix-recommender/
├── data/
│   └── Dataset.csv
├── notebook.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

```bash
git clone https://github.com/your-username/netflix-content-recommender.git
cd netflix-content-recommender
pip install -r requirements.txt
```

## Usage

Open `notebook.ipynb` and run the cells in order. To get recommendations for a specific title:

```python
get_recommendations('Sankofa', top_n=5)
```

## Results

Recommendations were validated against three titles spanning different genre combinations. All three returned a 100% genre-overlap score in the top 10 results, confirming the model correctly uses full multi-label genre combinations rather than clustering on a single dominant tag.

## Key limitations

- The dataset used does not include a plot description or cast column, so similarity relies primarily on genre and director rather than narrative content.
- Director names are tokenized individually, which can create incidental similarity between unrelated people who share a first or last name.
- The all-pairs similarity matrix is O(n²) in memory, which is fine at this dataset's scale (~8,800 titles) but would need a different approach (e.g. approximate nearest neighbors) at a much larger scale.

## Tech stack

Python, pandas, NumPy, scikit-learn (`CountVectorizer`, `cosine_similarity`)

## Author

Ocheme Vincent — Machine Learning Intern, Auspify Technology
