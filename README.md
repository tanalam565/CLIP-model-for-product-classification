# CLIP Model for Product Classification and Recommendation

This project explores how OpenAI CLIP can be used for product matching, recommendation, and zero-shot product classification on Amazon-style beauty and personal care data.

The repository contains two main notebooks:

- `product_similarity_amazon.ipynb` — builds CLIP text and image embeddings, compares query intent to product listings, and computes similarity scores.
- `zero shot testing_beauty_products.ipynb` — loads Amazon beauty product metadata, creates product embeddings, and tests zero-shot classification and retrieval behavior.

The test data in this repository is based on a new Amazon account search/recommendation set, which is stored in `Amazon_recom_queries.csv`.

## Project goal

The goal is to evaluate whether CLIP can:

- match a user search query to relevant Amazon products,
- rank products based on similarity between query text and product metadata/images,
- support zero-shot product classification using textual labels such as “matte lipstick,” “glossy lipstick,” or “long-lasting lipstick.”

## Data

### Amazon recommendation query file

The file `Amazon_recom_queries.csv` contains query-driven product recommendation examples. Each row represents one Amazon product candidate for a given query.

Columns include:

- `Queries` — the customer search phrase or product intent,
- `Product_link` — Amazon product URL,
- `Product_title` — product name,
- `Image_link` — product image URL,
- `Product_description` — product description and specifications.

This dataset was collected using a new Amazon account to simulate a fresh user account/test scenario rather than using historical account browsing behavior.

### Beauty product dataset

The notebooks also reference a beauty product dataset loaded from a Hugging Face Amazon metadata source and exported to a CSV named `beauty_products.csv`. This dataset contains product metadata and image links used to create CLIP embeddings for comparison and search.

## Methodology

The workflow in the notebooks follows these main steps:

1. Load product metadata and query records.
2. Download product images and metadata.
3. Generate CLIP embeddings for text and image content.
4. Normalize embeddings and compute cosine similarity.
5. Rank products for each query.
6. Evaluate zero-shot classification using category labels.

The architecture combines:

- CLIP text encoder for query and product text,
- CLIP image encoder for product image understanding,
- cosine similarity as the main comparison metric.

## Dependencies

The project uses Python packages such as:

- `torch`
- `torchvision`
- `Pillow`
- `pandas`
- `numpy`
- `datasets`
- `transformers`
- `sentence-transformers`
- `faiss-cpu`
- `requests`
- `scikit-learn`
- `openai/CLIP` via `clip`

A typical install command is:

```bash
pip install torch torchvision pillow sentence-transformers ftfy git+https://github.com/openai/CLIP.git pandas numpy datasets transformers faiss-cpu scikit-learn
```

## Notebook usage

### 1. Product similarity search

Open `product_similarity_amazon.ipynb` and run the cells in order:

- install dependencies,
- load the dataset,
- create text/image embeddings,
- compute similarity scores,
- review top matching products for each query.

### 2. Zero-shot product classification

Open `zero shot testing_beauty_products.ipynb` and run the cells in order:

- load CLIP model,
- prepare product metadata and image URLs,
- embed text and image content,
- evaluate query-to-product similarity,
- test zero-shot classification with labels such as product style, finish, or category.

## Example interpretation

For a query like:

- `Love Lipstick`

The model can compare the query text against product titles, descriptions, and images to find products that are visually and semantically similar. These results can then be used for ranking recommendation candidates or zero-shot labeling.

## Notes

- The repository is intended for research and experimentation rather than production deployment.
- The new Amazon account evaluation data is useful for testing search and recommendation behavior in a realistic but isolated setting.
- For better reproducibility, make sure the image URLs remain valid and that the dataset versions are consistent when rerunning the notebooks.

## Suggested next steps

- Add a clean evaluation table with precision@k and recall metrics.
- Expand the query set to include more categories beyond beauty products.
- Compare CLIP with a text-only or image-only baseline.
- Add a small recommendation dashboard or summary report.

## Repository structure

```text
CLIP-Model-for-Product-Classification/
├── README.md
├── Amazon_recom_queries.csv
├── product_similarity_amazon.ipynb
├── zero shot testing_beauty_products.ipynb
└── (optional generated CSVs such as beauty_products.csv)
```

This project is a practical CLIP-based demonstration of multimodal product matching for Amazon-like datasets and can be extended for larger recommendation or classification pipelines.
