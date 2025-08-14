Title: nomic-ai/modernbert-embed-base · Hugging Face

URL Source: http://huggingface.co/nomic-ai/modernbert-embed-base

Markdown Content:
[![Image 1: image/png](https://huggingface.co/nomic-ai/modernbert-embed-base/resolve/main/modernbertembed.png)](https://huggingface.co/nomic-ai/modernbert-embed-base)

ModernBERT Embed is an embedding model trained from [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base), bringing the new advances of ModernBERT to embeddings!

Trained on the [Nomic Embed](https://arxiv.org/abs/2402.01613) weakly-supervised and supervised datasets, `modernbert-embed` also supports Matryoshka Representation Learning dimensions of 256, reducing memory by 3x with minimal performance loss.

[](http://huggingface.co/nomic-ai/modernbert-embed-base#performance)Performance
-------------------------------------------------------------------------------

| Model | Dimensions | Average (56) | Classification (12) | Clustering (11) | Pair Classification (3) | Reranking (4) | Retrieval (15) | STS (10) | Summarization (1) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nomic-embed-text-v1 | 768 | 62.4 | 74.1 | 43.9 | **85.2** | 55.7 | 52.8 | 82.1 | 30.1 |
| nomic-embed-text-v1.5 | 768 | 62.28 | 73.55 | 43.93 | 84.61 | 55.78 | **53.01** | **81.94** | 30.4 |
| modernbert-embed-base | 768 | **62.62** | **74.31** | **44.98** | 83.96 | **56.42** | 52.89 | 81.78 | **31.39** |
| nomic-embed-text-v1.5 | 256 | 61.04 | 72.1 | 43.16 | 84.09 | 55.18 | 50.81 | 81.34 | 30.05 |
| modernbert-embed-base | 256 | 61.17 | 72.40 | 43.82 | 83.45 | 55.69 | 50.62 | 81.12 | 31.27 |

[](http://huggingface.co/nomic-ai/modernbert-embed-base#usage)Usage
-------------------------------------------------------------------

You can use these models directly with the latest transformers release and requires installing `transformers>=4.48.0`:

```
pip install transformers>=4.48.0
```

Reminder, this model is trained similarly to Nomic Embed and **REQUIRES** prefixes to be added to the input. For more information, see the instructions in [Nomic Embed](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5#task-instruction-prefixes).

Most use cases, adding `search_query: ` to the query and `search_document: ` to the documents will be sufficient.

### [](http://huggingface.co/nomic-ai/modernbert-embed-base#sentence-transformers)Sentence Transformers

```
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/modernbert-embed-base")

query_embeddings = model.encode([
    "search_query: What is TSNE?",
    "search_query: Who is Laurens van der Maaten?",
])
doc_embeddings = model.encode([
    "search_document: TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten",
])
print(query_embeddings.shape, doc_embeddings.shape)
# (2, 768) (1, 768)

similarities = model.similarity(query_embeddings, doc_embeddings)
print(similarities)
# tensor([[0.7214],
#         [0.3260]])
```

Click to see Sentence Transformers usage with Matryoshka TruncationIn Sentence Transformers, you can truncate embeddings to a smaller dimension by using the `truncate_dim` parameter when loading the `SentenceTransformer` model.

```
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nomic-ai/modernbert-embed-base", truncate_dim=256)

query_embeddings = model.encode([
    "search_query: What is TSNE?",
    "search_query: Who is Laurens van der Maaten?",
])
doc_embeddings = model.encode([
    "search_document: TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten",
])
print(query_embeddings.shape, doc_embeddings.shape)
# (2, 256) (1, 256)

similarities = model.similarity(query_embeddings, doc_embeddings)
print(similarities)
# tensor([[0.7759],
#         [0.3419]])
```

Note the small differences compared to the full 768-dimensional similarities.

### [](http://huggingface.co/nomic-ai/modernbert-embed-base#transformers)Transformers

```
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


queries = ["search_query: What is TSNE?", "search_query: Who is Laurens van der Maaten?"]
documents = ["search_document: TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten"]

tokenizer = AutoTokenizer.from_pretrained("nomic-ai/modernbert-embed-base")
model = AutoModel.from_pretrained("nomic-ai/modernbert-embed-base")

encoded_queries = tokenizer(queries, padding=True, truncation=True, return_tensors="pt")
encoded_documents = tokenizer(documents, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    queries_outputs = model(**encoded_queries)
    documents_outputs = model(**encoded_documents)

query_embeddings = mean_pooling(queries_outputs, encoded_queries["attention_mask"])
query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
doc_embeddings = mean_pooling(documents_outputs, encoded_documents["attention_mask"])
doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)
print(query_embeddings.shape, doc_embeddings.shape)
# torch.Size([2, 768]) torch.Size([1, 768])

similarities = query_embeddings @ doc_embeddings.T
print(similarities)
# tensor([[0.7214],
#         [0.3260]])
```

Click to see Transformers usage with Matryoshka TruncationIn `transformers`, you can truncate embeddings to a smaller dimension by slicing the mean pooled embeddings, prior to normalization.

```
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


queries = ["search_query: What is TSNE?", "search_query: Who is Laurens van der Maaten?"]
documents = ["search_document: TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten"]

tokenizer = AutoTokenizer.from_pretrained(".")
model = AutoModel.from_pretrained(".")
truncate_dim = 256

encoded_queries = tokenizer(queries, padding=True, truncation=True, return_tensors="pt")
encoded_documents = tokenizer(documents, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    queries_outputs = model(**encoded_queries)
    documents_outputs = model(**encoded_documents)

query_embeddings = mean_pooling(queries_outputs, encoded_queries["attention_mask"])
query_embeddings = query_embeddings[:, :truncate_dim]
query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
doc_embeddings = mean_pooling(documents_outputs, encoded_documents["attention_mask"])
doc_embeddings = doc_embeddings[:, :truncate_dim]
doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)
print(query_embeddings.shape, doc_embeddings.shape)
# torch.Size([2, 256]) torch.Size([1, 256])

similarities = query_embeddings @ doc_embeddings.T
print(similarities)
# tensor([[0.7759],
#         [0.3419]])
```

Note the small differences compared to the full 768-dimensional similarities.

### [](http://huggingface.co/nomic-ai/modernbert-embed-base#transformersjs)Transformers.js

If you haven't already, you can install the [Transformers.js](https://huggingface.co/docs/transformers.js) JavaScript library from [NPM](https://www.npmjs.com/package/@huggingface/transformers) using:

```
npm i @huggingface/transformers
```

Then, you can compute embeddings as follows:

```
import { pipeline, matmul } from '@huggingface/transformers';

// Create a feature extraction pipeline
const extractor = await pipeline(
  "feature-extraction",
  "nomic-ai/modernbert-embed-base",
  { dtype: "fp32" }, // Supported options: "fp32", "fp16", "q8", "q4", "q4f16"
);

// Embed queries and documents
const query_embeddings = await extractor([
    "search_query: What is TSNE?",
    "search_query: Who is Laurens van der Maaten?",
  ], { pooling: "mean", normalize: true },
);
const doc_embeddings = await extractor([
    "search_document: TSNE is a dimensionality reduction algorithm created by Laurens van Der Maaten",
  ], { pooling: "mean", normalize: true },
);

// Compute similarity scores
const similarities = await matmul(query_embeddings, doc_embeddings.transpose(1, 0));
console.log(similarities.tolist()); // [[0.721383273601532], [0.3259955644607544]]
```

[](http://huggingface.co/nomic-ai/modernbert-embed-base#training)Training
-------------------------------------------------------------------------

Click the Nomic Atlas map below to visualize a 5M sample of our contrastive pretraining data!

[![Image 2: image/webp](https://cdn-uploads.huggingface.co/production/uploads/607997c83a565c15675055b3/pjhJhuNyRfPagRd_c_iUz.webp)](https://atlas.nomic.ai/map/nomic-text-embed-v1-5m-sample)

We train our embedder using a multi-stage training pipeline. Starting from a long-context [BERT model](https://huggingface.co/nomic-ai/nomic-bert-2048), the first unsupervised contrastive stage trains on a dataset generated from weakly related text pairs, such as question-answer pairs from forums like StackExchange and Quora, title-body pairs from Amazon reviews, and summarizations from news articles.

In the second finetuning stage, higher quality labeled datasets such as search queries and answers from web searches are leveraged. Data curation and hard-example mining is crucial in this stage.

For more details, see the Nomic Embed [Technical Report](https://static.nomic.ai/reports/2024_Nomic_Embed_Text_Technical_Report.pdf) and corresponding [blog post](https://blog.nomic.ai/posts/nomic-embed-text-v1).

Training data to train the models is released in its entirety. For more details, see the `contrastors` [repository](https://github.com/nomic-ai/contrastors)

[](http://huggingface.co/nomic-ai/modernbert-embed-base#join-the-nomic-community)Join the Nomic Community
---------------------------------------------------------------------------------------------------------

*   Nomic: [https://nomic.ai](https://nomic.ai/)
*   Discord: [https://discord.gg/myY5YDR8z8](https://discord.gg/myY5YDR8z8)
*   Twitter: [https://twitter.com/nomic\_ai](https://twitter.com/nomic_ai)

[](http://huggingface.co/nomic-ai/modernbert-embed-base#citation)Citation
-------------------------------------------------------------------------

If you find the model, dataset, or training code useful, please cite our work

```
@misc{nussbaum2024nomic,
      title={Nomic Embed: Training a Reproducible Long Context Text Embedder}, 
      author={Zach Nussbaum and John X. Morris and Brandon Duderstadt and Andriy Mulyar},
      year={2024},
      eprint={2402.01613},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```