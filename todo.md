Todo:
[x] Ingestion of documents
[x] Summary Generation. This uses a config file at `config/config.json` to specify model and parameters.
[x] Chunk Generation. Here, we want to chunk while respecting sentence boundaries as best as we can. Let's start by using regex. All a "test" mode via a --test flag where I can inspect a few chunks in the terminal. There's a min chunk length - in characters - and max also that should be specified in the config file. Some consideration should be given to how csvs and tables (in markdown in text files) are handled neatly. We should also have a last resort fallback of relying maybe on character count.
[x] Q&A Generation - single chunk. Specify what llm is going to be used in the config file. We'll generate questions for each document, i.e. iterate through each document. For each document, we'll send a separate request to the LLM for each chunk and ask for question, evaluation criteria, reasoning behind the answer, answer, difficulty on a scale of 1 to 10, question category and citations (drawing verbatim from the chunk provided). The LLM should respond by first providing it's thought process on the task at hand - within <think></think> tags, and then respond with a pydantic object for each quesiton answer pair. Provide the llm with instructions and a sample response with the correct format in the system message. Provide the document summary and chunk in a first user message. The LLM should create as many pydantic object responses as it deems necessary to capture the key information within the chunk. Allow for a test mode via a --test flag, whereby questions are only generated for a first chunk. The summary is provided to contextualise the chunk. The questions should be educational and span abilities from novice to expert. The quesitons should span a range of categories necessary to deeply comprehend the material described within the chunk.
[x] Create a training dataset and push to huggingface (note the user will have to log in) - and search how best to do this. Allow the user's org to be set in the config file. The qa dataset should be pushed with a subset of 'qa' and using fields matching those of the pydantic object. There should be a config file option to push as public or private. Push the chunks as another subset and push the raw text as another subset.
---
[x] Adjust test mode for qa to generate for first two chunks.
[x] Manually review the qa prompt and test with pro vs flash model. Flash model gives 11/11 good questions. Pro model gives 10/10 good questions. Sonnet 3.5 gives 6 questions but the thinking portion wrongly justifies what the question aims to test, not the reasoning for the answer.
[-] Support LiteLLM. I should really be using litellm instead throughout the repo as that has consistent handling of both inputs and outputs. Look up litellm docs for google/gemini and also anthropic and update my repo accordingly. NOT NEEDED, I MISUNDERSTOOD OPENROUTER, which is consistent with outputs.
---
[x] Initial ablations:
    [x] How do I measure the difficulty of questions? Round-robin the testing, see if they perform the same on their own questions or not. Need to hold judge constant and have good enough repeatability on question sets regenerated with sampling.
    [x] Are question difficulties consistent when repeating the same qa generation? i.e. generate two datasets the same way and measure the evaluation. **They seem to be roughly similar, within a few percentage points**
    [x] Do more powerful models ask harder questions? **Maybe a little bit, not statisticall yet**
    [x] Do questions get harder if using larger chunks - increase from 5k max characters to 20k? Does the number of total questions go up or down? Total questions goes down (e.g. 48 down from ~70). **Not obviously so, but you get less questions generated.**
---
[x] Cleanup
    [x] Remove the reasoning column from the dataset creation script.
    [x] Remove the think tags usage from the answers, to simplify. Include reasoning steps in the criteria, if appropriate. Allow for the possibility that there may be various paths to a correct answer.
    [x] Tidy up scripts into a utils folder.
    [x] Tidy up configs and allow any script to be called by passing a config file path, otherwise default to using `config/config.json`.
---
[x] Clustering and visualisation.
    [x] **Do similar questions cluster together in embedding space? How does this look?** If I re-run question generation, how similar are the questions generated? Examine the similarity of each question in the evaluation set with the closest question in the training set. Distant questions suggest insufficient training set coverage. Near identical questions indicate either mode collapse or insufficient temperature variation. I want to visualise this - probably by collapsing embeddings down onto a 2D space. To compute embeddings probably use batch mode on this model: https://huggingface.co/nomic-ai/modernbert-embed-base, run locally.
    [x] **Implementation Plan:**
      [x] Enhance visualization tool to include cluster analysis (elbow method) for determining optimal cluster count
      [x] Create a new script to iteratively build dataset with smart sampling and deduplication based on embedding similarity
      [x] Modify create_dataset.py to enable stratified eval set creation based on silhouette analysis
      [x] Allow visualisation of dataset splits.
---
[x] Fine-tuning for memorization - ATTEMPT 1:
  [x] Fine-tune Gemma:
    [x] Comprehensive/balanced dataset vs naive dataset (no dedup but with augmented answers):
      [x] Does fine-tuning improve performance on the eval set? Only a little, likely the deduplication threshold is too strict and knowledge is omitted.
      [x] Does the eval loss drop? Yes, drops a lot.
      - Broadly, the performance is worse than a naive dataset with augmented answers (criteria + step by step). THEREFORE - THERE IS A NEED TO INSPECT HOW THE DATA IS CREATED 
---
[x] Data visualisation.
  [x] Inspect all questions generated for a chunk by a certain model. Do the questions asymptote?
    [x] Read the questions
    [x] Visualise the questions (combine the visualisation and embedding scripts into one, and allow for multiple datasets).
  [x] Do different models generate different questions - as per the visualisation? Broadly, the models seem somewhat similar.
  [x] Create a tag-based visualisation tool that uses gemini flash to generate tags for each question and then create an interactive plot that allows multiple datasets to be compared - it should be possible to see tag-based cluster names, and also view individual questions when hovering over them. Do this with two scripts, one to create tags and then one to compare_tags.py
  [x] Is the naive synth dataset or the comprehensive dataset more difficult? NOTE THEY ARE NOT EXACTLY COMPARABLE AS THE NAIVE DATASET IS SCORED VIA GROUND TRUTH. The comprehensive dataset is scoring 30% while the naive one is scoring 40%.
  [x] Does a comprehensive gemini pro dataset encapsulate the naive dataset? If yes, then proceed. Result = they seem somewhat similar although the skew is a bit different.
---
[x] Fine-tuning for memorization - Attempt 2 (now with a more complete dataset):
  [x] Train on comprehensive dataset and compare performance on synth eval split. Aiming to be as good if not better. A model fine-tuned on the naive synth train split performs better than fine-tuned on comprehensive data on the naive synth eval split - this suggests either a) contamination in that dataset between train and eval, b) better answers, c) better question distribution.
  - Note that the training is poor enough on the 1B parameter model as it goes from 6% up to about 15%, not great. Perhaps the learning rate and or batch size is too low as the training loss is still falling. But the naive synth dataset only gets to ~12%, which is lower but it doesn't have the advantage of training on that same set.
  [x] Try testing both models on my manual touch rugby dataset of just five rows. See how both perform. SEEMS THE COMPREHENSIVE SET IS BETTER THAN THE NAIVE SET, BUT THE ORIGINAL MODEL IS BETTER THAN BOTH.
---
[x] Improve the manual touch rugby dataset by adding rows. NOTE THAT THE CURRENT TRAINING IS UNDERPERFORMING THE BASE MODEL (the ft model on naive data is even worse). I also see that the quality of questions from gemini pro is not the best because some questions are not well contextualised.
---
[x] Eval split creation:
  [x] Rephrase the questions to create an eval split. THIS CAN NOW BE USED FOR TRAINING.
  - Mirror results are a little higher than re-phrased questions, indicating generalisation to rephrasing is good.
---
[x] Improve fine-tuning - Part 2:
  [x] Adjust the LR.
  [x] Measure performance on the training set AND on the eval dataset. Adjust the notebook to allow for this. EVAL SET PERFORMANCE IS IMPROVING, BUT MANUAL DATASET PERFORMANCE IS STILL POOR... AT LEAST ON 1B MODEL. CAN TRY OUT GEMMA MODEL NEXT.
  [x] Inspect logs with wandb (esp. grad norm). PERHAPS I NEED TO ALSO RE-WORD THE ANSWERS AS WELL AS THE QUESTIONS, BECAUSE THE EVAL LOSS IS CLOSELY MIRRORING THE TRAINING LOSS, AND NOT GIVING MUCH ADDED INFO to tell if we are overfitting or not. UPDATE THE EVAL LOSS WAS NOT PROPERLY SET, IT IS IN FACT A GOOD INDICATOR.
  [x] Run evals on the answers from the fine-tuning dataset, to check the criteria. ALL 32 MIRRORED ANSWERS ARE MARKED CORRECT. All 32 (rephrased) eval q&a pairs and 242/244 training answers are marked correct. VERY GOOD. JUDGING IS ALMOST PERFECT.
  [x] Try re-wording the questions as well as the answers.
    [x] Re-train to see loss curves and if more meaningful. Yes eval loss now makes more sense to monitor alright. And yes, even the modified version is useful. YES, ACTUALLY YOU DO SEE DIVERGENCE IN THE MIRROR AND EVAL SPLITS IF YOU ARE OVERFITTING. MANUAL PERFORMANCE IS SOMEWHAT LIKE THE EVAL PERFORMANCE (THE REPHRASED).
---
[x] Check contextualisation of questions from different models. Using the eval tool. Seems to be ok but not perfect for Gemini Pro 2.5, a few rows have no reference to touch rugby.
---
[x] Final fine-tuning - on Gemma - Test out Gemma model on fine-tuning to see performance we can get with a 4B model.
  [x] Check unsloth notebook for training on completions only.
  [x] Train on the comprehensive dataset.
    [ ] Testing Gemma. there is an issue with vLLM that requires resolution.
  [ ] Train on the naive dataset. DEFERRED.
---
[ ] Fine-tuning for reasoning - answer improvements:
  [ ] Does just asking to answer step by step improve performance?
  [ ] Evaluate whether "augmentation" (criteria + step by step) is effective in improving performance.
  [ ] Evaluate whether using a reasoning model - and including those traces - improves performance, i.e. DeepSeek.
---
[ ] Multi-chunk reasoning. This should mirror single chunk reasoning. However, we'll pass in between min_chunks and max_chunks number of chunks at once. We should also let the LLM know to only generate questions rely on all chunks provided in context. If there are no further sensible questions to create, do not return any further pydantic objects. It may be acceptable to return no qa pairs - in which case we should handle that gracefully. Multi-chunk groups should be created by iterating from k starting at min_chunks up to max_chunks, and finding all possible combinations of chunks that can be grouped together for that value of k. A field should also be recorded so the number of chunks will be known at dataset creation time.
  [ ] Are the questions generated harder for more chunks?
    [ ] As rated by difficulty?
    [ ] As measured by model performance?
    [ ] Do I get more reasoning questions?
  [ ] Is there a shortcut to deciding what groups to consider?
    [ ] Are more questions generated for similar chunks or dissimilar chunks?
    [ ] Are more difficult questions generated for similar chunks or dissimilar chunks?
---
[ ] Reinforcement Learning:
    [ ] Does RL improve performance on reasoning questions?
---
For later:
[ ] Chunk Generation
    [ ] Review csv performance.
    [ ] Make chunking be section/heading-aware.
[ ] Question Generation
    [ ] Measure coverage.
    [ ] Measure difficulty.
    [ ] Measure contextualisation.
    [ ] Support passing of images to the LLM.
    [ ] Incorporate citation verification.
[ ] ADVANCED-evals repo
  [ ] Allow evaluation on a runpod endpoint, in evals. WILL ALLOW FOR THIS LATER, perhaps via HF API.