Key learnings:
- EMPIRICAL - LLM fine-tuning is reasonably robust to rewording questions (and maybe answers too?) --- PROVIDED YOU DON'T OVERFIT (I.E. RISING EVAL LOSS).
- THEORETICAL - If you just pick a random subset for eval, you risk having an unrepresentative sample if there are many groups of question types in your data.
- EMPIRICAL - tag based classification and embedding based classification end up pretty similar in terms of identifying whether datasets overlap or not.
- THEORETICAL - larger chunks have better context.
- THEORETICAL - non-uniform chunks better respect context but risk non-uniform question generation, depending on how you create questions.
- EMPIRICAL - using larger batch sizes reduces grad norm, allowing for a higher learning rate.