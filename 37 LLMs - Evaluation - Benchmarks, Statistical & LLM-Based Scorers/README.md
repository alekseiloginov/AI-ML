LLMs - Evaluation - Benchmarks, Statistical & LLM-Based Scorers

Models we're using are from Hugging Face:
- Mistral 7B: https://huggingface.co/mistralai/Mistral-7B-v0.1
- Distil GPT-2: https://huggingface.co/distilbert/distilgpt2

Goals:
- Evaluate the model using benchmarks, statistic-based, and model-based approaches.
- Evaluate truthfulness of answers using `TruthfulQA` benchmark.
- Evaluate expert knowledge using `MMLU (Massive Multitask Language Understanding)` benchmark.
- Evaluate code-generation abilities using `HumanEval` benchmark.
- Evaluate automatic summarization and machine translation using `ROUGE` metrics.
- Evaluate machine translation using `METEOR` metric.
- Evaluate factual correctness using `G-Eval` framework.
- Evaluate embeddings-based similarity using `BERTScore` approach.
- Evaluate hallucinations using `SelfCheckGPT` approach.
- Evaluate the evaluation methods using `Kendall-Tau` and `Spearman` correlation coefficients.
