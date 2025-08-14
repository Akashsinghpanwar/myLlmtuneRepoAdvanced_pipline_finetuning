# Data Preparation

Generate high quality synthetic datasets for evaluation or fine-tuning.

## Document Ingestion

### Setup

- Ensure you have [uv](https://github.com/astral-sh/uv) installed.
- Install dependencies (marker for PDF extraction and pypandoc for DOCX/DOC conversion) - NOT NECESSARY IF GIT CLONING:
```sh
uv init
rm main.py
uv add "marker-pdf"
uv add "pypandoc"
```

### Usage

1. Place your files in `data/raw/` (supports: pdf, docx, doc, md, txt, py, csv, json).
2. Run the ingestion script from the `reasoning` directory:

```sh
uv run utils/ingest.py
```

- By default, only new files (not yet processed) will be ingested.
- To re-ingest all files (overwrite outputs), use:

```sh
uv run utils/ingest.py --force
```

3. Extracted text will be saved in `data/text/`, one `.txt` file per input.

## Summary Generation

### Setup

1. Install dependencies:

```sh
uv add requests python-dotenv
```

2. Create a `.env` file in the `reasoning` directory with your OpenRouter API key:

```
OPENROUTER_API_KEY=your_api_key_here
```

You can use the provided `.env.example` file as a template. You'll need to sign up at [OpenRouter](https://openrouter.ai/) to get your API key.

3. Configure the summarization settings in `config/config.json` (or pass `--config <path>` to override):

```json
{
  "summary": {
    "model": "google/gemini-2.0-flash-001",
    "temperature": 0.3,
    "max_tokens": 1000
  }
}
```

If the file doesn't exist, it will be created automatically with default values.

### Usage

Run the summary generation script from the `reasoning` directory:

```sh
uv run utils/summarize.py
```

By default, this will:
- Use the model specified in the config.json file (defaults to Gemini Flash 2.0)
- Generate summaries only for files that haven't been summarized yet
- Save summaries as JSON files in `data/summaries/`

Options:
- To regenerate all summaries (overwrite existing ones), use:
  ```sh
  uv run utils/summarize.py --force
  ```
- To override the model specified in `config/config.json`, use the `--model` flag and/or specify a different config via `--config`:
  ```sh
  uv run utils/summarize.py --model "google/gemini-2.0-flash-001" --config config/config.json
  ```

## Chunk Generation

### Setup

- Install dependencies:

```sh
uv add regex
```

### Usage

1. Run the chunk generation script from the `reasoning` directory:

```sh
uv run utils/chunk.py
```

- By default, this will:
  - Use regex to chunk the text while respecting sentence boundaries
  - Save chunks as JSON files in `data/chunks/`
- Options:
  - To override the default chunking settings, use the `--config` flag:
    ```sh
    uv run utils/chunk.py --config config/config.json
    ```
  - To inspect a few chunks in the terminal, use the `--test` flag:
    ```sh
    uv run utils/chunk.py --test
    ```

## Q&A Generation

The QA generation system creates high-quality question-answer pairs from document chunks using LLMs.

### Features
- Generates questions and answers with detailed evaluation criteria
- Supports parallel processing for faster generation
- Handles various document formats and chunk sizes
- Configurable model parameters and generation settings

### Configuration
In your `config/config.json` file, you can configure the QA generation process:

```json
"qa": {
  "model": "google/gemini-2.0-flash-001",
  "temperature": 0.7,
  "max_tokens": 16384,
  "top_p": 0.95,
  "top_k": 40,
  "min_p": 0.05,
  "use_batching": true,
  "batch_size": 32,
  "skip_tables": false,
  "max_iterations": 3
}
```

### Parallel Processing
The QA generation now supports parallel processing to significantly speed up question generation:
- `use_batching`: Set to `true` to enable parallel processing (default: true)
- `batch_size`: Number of concurrent API requests (default: 32)

### Iterative Question Generation
The QA generation system now supports iterative question generation to ensure comprehensive coverage of concepts:
- `--iterate`: Enable iterative question generation for each chunk
- `max_iterations`: Maximum number of iterations per chunk (default: 3)

When iterative mode is enabled, the system will:
1. Generate an initial set of questions for a chunk
2. In subsequent iterations, provide the previously generated questions to the model
3. Ask the model to generate additional questions focusing on uncovered aspects
4. Continue until either no new questions are generated or max_iterations is reached
5. Save both individual iteration results and a combined result with all questions

### Usage
```bash
uv run utils/generate_qa.py [options]
```

### Options
- `--config`, `-c`: Path to config file
- `--force`: Regenerate Q&A pairs even if already processed
- `--test`: Test mode (only process first two chunks)
- `--doc`: Process only the specified document
- `--iterate`: Enable iterative question generation for each chunk

## Iterative QA Generation (OPTIONAL ALTERNATIVE)

>[!TIP]
>This is alternative method where questions are generated for all chunks. Then that same process is re-run and only new (dissimilar) questions are kept and added to the dataset. This does generate a larger variety of questions but is less efficient than the basic script (which feeds in already generated questions) because it will result in generating lots of questions that were already generated.

The iterative dataset generation system builds high-quality datasets by generating questions across multiple iterations with smart sampling and deduplication.

### Features
- Generates fresh questions in each iteration using iteration-specific directories
- Deduplicates questions based on embedding similarity (using modernbert-embed-base)
- Tracks acceptance rates and coverage gains to determine when to stop
- Automatically stops when new data provides diminishing returns
- Supports resuming from previous iterations

### Usage
```bash
uv run utils/build_dataset_iterative.py [options]
```

### Options
- `--config`, `-c`: Path to config file
- `--doc`, `-d`: Specific document to process
- `--max-iterations`, `-m`: Maximum number of iterations (default: 5)
- `--test`, `-t`: Run in test mode (processes only first 2 chunks)
- `--batch-size`, `-b`: Batch size for each iteration (default: 100)
- `--similarity-threshold`, `-s`: Similarity threshold for deduplication (default: 0.92)
- `--min-acceptance-rate`, `-a`: Minimum acceptance rate to continue (default: 0.2)
- `--min-coverage-gain`, `-g`: Minimum coverage gain rate to continue (default: 0.05)
- `--force`, `-f`: Force regeneration of data even if it exists

### Stopping Criteria
The process stops when either:
1. The acceptance rate drops below the minimum threshold (default: 20%)
2. The coverage gain over recent iterations drops below the minimum (default: 5%)
3. The maximum number of iterations is reached

### Resuming Process
By default, the system will:
1. Check for existing iteration data
2. Load all previously generated and deduplicated questions
3. Resume from the next iteration
4. Use the `--force` flag to start from scratch and regenerate all data

## Dataset Creation and Hugging Face Upload

### Setup

1. Install dependencies:

```sh
uv add datasets huggingface-hub sentence-transformers scikit-learn matplotlib kneed
```

2. Configure the dataset settings in `config/config.json`:

```json
{
  "dataset": {
    "name": "touch-rugby-reasoning",
    "organization": "Trelis",
    "public": true,
    "seed": 42
  }
}
```

3. If not already logged in to Hugging Face Hub, the script will prompt you to log in during execution. You can also manually log in using:

```sh
uv run -m huggingface_hub login
```

### Creating a Dataset with Stratified Evaluation Split

The dataset creation tool now supports creating stratified evaluation splits based on clustering analysis, ensuring balanced representation of different question types in both training and evaluation sets.

```bash
# Create dataset from default QA directory
uv run utils/create_dataset.py --config config/config.json --eval-split

# Create dataset from iterative dataset builder output
uv run utils/create_dataset.py --config config/config.json --eval-split --input-dir data/final_dataset
```

### Options
- `--config`, `-c`: Path to config file
- `--no-push`: Don't push the dataset to Hugging Face Hub
- `--eval-split`: Create a stratified train/eval split
- `--eval-ratio`: Ratio of data to use for evaluation (default: 0.2)
- `--input-dir`: Custom input directory for QA data (e.g., path to final_dataset from iterative builder)

### Advanced Evaluation Split Features

The dataset creation script now supports advanced evaluation split creation:

1. **Smart Sampling**: When using `--eval-split`, the script will:
   - Select up to 20% of the dataset (or 32 examples, whichever is lower)
   - Sample proportionally from each cluster identified by the elbow method
   - Keep the selected examples in the training split for better coverage

2. **Evaluation Mirror**: Create a mirror of the evaluation split with identical examples:
```sh
uv run utils/create_dataset.py --eval-split --eval-mirror
```
   This is useful for comparing different evaluation methods on the same examples.

3. **Question and Answer Rephrasing**: Create an evaluation split with rephrased questions and answers:
   ```sh
   uv run utils/create_dataset.py --eval-split --rephrase-qa
   ```
   This uses the google/gemini-2.0-flash-001 model via OpenRouter to rephrase both questions and answers while maintaining their meaning.
   The rephrasing happens in a single API call for each question-answer pair, ensuring consistency between the rephrased question and answer.
   Requires an OpenRouter API key in the `.env` file (OPENROUTER_API_KEY).

You can combine these options:
```sh
uv run utils/create_dataset.py --eval-split --eval-mirror --rephrase-qa
```

### Stratified Split Process

The stratified split process works as follows:

1. Embeddings are generated for all questions using modernbert-embed-base
2. Optimal number of clusters is determined using the elbow method:
   - Calculates inertia (sum of squared distances) for different cluster counts
   - Automatically detects the "elbow point" where adding more clusters gives diminishing returns
3. Questions are clustered using K-means with the optimal number of clusters
4. A balanced sample of up to 20% (or max 32 examples) is selected proportionally from each cluster
5. For rephrased questions and answers, the original meaning is preserved while changing the wording

## Embedding Visualization

The `visualisation` directory contains tools for visualizing and analyzing embeddings. These tools allow you to:

1. Generate embeddings for questions in datasets
2. Visualize embeddings within a single dataset
3. Visualize train vs. eval splits to assess your stratified splitting strategy
4. Compare embeddings between up to three different datasets

For detailed documentation on using these visualization tools, please refer to the [visualisation README](./visualisation/README.md).