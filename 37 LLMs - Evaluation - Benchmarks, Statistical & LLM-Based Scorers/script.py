import nltk
import hugging_face_model
import scipy.stats as stats
import gemini
from deepeval.benchmarks import TruthfulQA, MMLU, HumanEval
from deepeval.benchmarks.tasks import TruthfulQATask, MMLUTask, HumanEvalTask
from deepeval.benchmarks.modes import TruthfulQAMode
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from rouge import Rouge
from nltk.translate import meteor_score
from evaluate import load
from selfcheckgpt.modeling_selfcheck import SelfCheckLLMPrompt

### Create a model which we will be evaluating

## Hugging Face models
# model_name = "distilbert/distilgpt2"  # fast, but very simple
# model_name = "mistralai/Mistral-7B-v0.1"  # smart, but slow and requires a HF access token
# model = hugging_face_model.new_model(model_name)

## Gemini model
model_name = "gemini-1.5-pro-001"
model = gemini.new_model(model_name)


### BENCHMARKS

## 1. TruthfulQA: truthfulness of answers
benchmark = TruthfulQA(
    tasks=[TruthfulQATask.STATISTICS],
    mode=TruthfulQAMode.MC2
)
benchmark.evaluate(model=model)
print('TruthfulQA score: ', benchmark.overall_score)

## 2. MMLU (Massive Multitask Language Understanding): expert knowledge
benchmark = MMLU(
    tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE],
    n_shots=3
)
benchmark.evaluate(model=model)
print('MMLU score: ', benchmark.overall_score)

## 3. HumanEval: code-generation abilities
benchmark = HumanEval(
    tasks=[HumanEvalTask.SORT_NUMBERS],
    n=100
)
benchmark.evaluate(model=model, k=10)
print('HumanEval score: ', benchmark.overall_score)


### Statistical Scorers

## 1. ROUGE: metric for automatic summarization and machine translation
# Sample reference and generated summaries
reference_sentence = 'A fast brown fox jumps over a lazy dog.'
generated_translation = model.generate("Translate to Russian: %s" % reference_sentence)
generated_sentence = model.generate("Translate to English: %s" % generated_translation)
print(generated_sentence)
# Calculate ROUGE scores
rouge_score = Rouge().get_scores([reference_sentence], [generated_sentence])
print('ROUGE Score:', rouge_score)

## 2. METEOR: advanced metric for machine translation
nltk.download('wordnet')
# Sample reference and generated sentences
reference_sentences = 'A fast brown fox jumps over a lazy dog.'
generated_translation = model.generate("Translate to Russian: %s" % reference_sentence)
generated_sentence = model.generate("Translate to English: %s" % generated_translation)
print(generated_sentence)
# Calculate METEOR score
meteor_score = meteor_score.meteor_score([reference_sentences.split()], generated_sentence.split())
print('METEOR Score:', meteor_score)


### LLM-Based Scorers

## 1. G-Eval: evaluate LLM outputs using custom requests to LLMs (a.k.a. LLM-Evals)
correctness_metric = GEval(
    name="Correctness",
    criteria="Determine whether the actual output is factually correct based on the expected output.",
    # NOTE: you can only provide either criteria or evaluation_steps, and not both
    # evaluation_steps=[
    #     "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
    #     "You should also heavily penalize omission of detail",
    #     "Vague language, or contradicting OPINIONS, are OK"
    # ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=model,
)
input = "The dog chased the cat up the tree, who ran up the tree?"
expected_output = "The cat."
actual_output = "It depends, some might consider the cat, while others might argue the dog."
actual_output = model.generate(input)
print(actual_output)
test_case = LLMTestCase(
    input=input,
    expected_output=expected_output,
    actual_output=actual_output,
)
correctness_metric.measure(test_case)
g_eval_score = correctness_metric.score
print(g_eval_score)
print(correctness_metric.reason)

## 2. BERTScore: pairwise cosine similarity based on token-level embeddings
input = "Who designed the Eiffel Tower?"
reference = "Gustave Eiffel"
prediction = model.generate(input)
print(prediction)
bert_score = load("bertscore")
results = bert_score.compute(predictions=[prediction], references=[reference], lang="en")
print(results)

## 3. SelfCheckGPT: reference-less sampling-based approach to detect hallucinations (fact-check LLM outputs)
input = "Generate two sentences about Albert Einstein's wives and mistresses"
text = model.generate(input)
# split text into sentences
nltk.download('punkt')  # Download necessary data (only once)
sentences = nltk.sent_tokenize(text)
print(sentences)
# Other samples generated by the same LLM to perform self-check for consistency
sample1 = model.generate(input)
print(sample1)
sample2 = model.generate(input)
print(sample2)
sample3 = model.generate(input)
print(sample3)
# perform self check
llm_model = "mistralai/Mistral-7B-v0.1"
device = "cpu"
self_check_prompt = SelfCheckLLMPrompt(llm_model, device)
sentences_scores_prompt = self_check_prompt.predict(
    sentences = sentences,                          # list of sentences
    sampled_passages = [sample1, sample2, sample3], # list of sampled passages
    verbose = True, # whether to show a progress bar
)
print(sentences_scores_prompt) # higher score indicates higher chance of being hallucination


### Evaluating the Evaluation Methods

## 1. Kendall's correlation coefficient: tau (𝜏)
# use correctness of a model on different text summaries
rouge_l_scores = [0.7, 0.8, 0.65, 0.75]
actual_scores = [0.8, 0.75, 0.7, 0.85]
kendall_tau, _kendall_p_value = stats.kendalltau(rouge_l_scores, actual_scores)
print("Kendall's tau:", kendall_tau)  # score of 0.7+ is generally regarded as good enough

## 2. Spearman's rank correlation coefficient: rho (ρ)
# use factual correctness of a model on different questions
g_eval_scores = [0.8, 0.75, 0.7, 0.85]
actual_scores = [0.95, 0.8, 0.9, 0.75]
spearman_rho, _spearman_p_value = stats.spearmanr(g_eval_scores, actual_scores)
print("Spearman's rho:", spearman_rho)
