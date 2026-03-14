# Personality AI — Psycholinguistic Personality Prediction

A machine learning system that predicts **Big Five personality traits (OCEAN)** from written text using **psycholinguistic linguistic features and interpretable machine learning models**.

This project explores the intersection of:

- Natural Language Processing (NLP)
- Computational Psychology
- Psycholinguistics
- Explainable Machine Learning

The model analyzes linguistic patterns in text and predicts the probability of each personality trait.

Note: The dataset to train can be found in the following link

```bash
https://huggingface.co/datasets/jingjietan/essays-big5/tree/main/data
```
---

# Big Five Personality Model

The system predicts the **OCEAN model** of personality:

| Trait | Description |
|-----|-----|
| **O** | Openness to Experience |
| **C** | Conscientiousness |
| **E** | Extraversion |
| **A** | Agreeableness |
| **N** | Neuroticism |

The output is a **probability score for each trait**.

Example output:


O: 0.46
C: 0.49
E: 0.37
A: 0.64
N: 0.49


---

# Project Pipeline

The system follows the pipeline:


Text
↓
Preprocessing
↓
Psycholinguistic Feature Extraction
↓
Feature Vector (25 dimensions)
↓
Random Forest Classifiers
↓
Personality Prediction


---

# Mathematical Formulation

The model treats personality prediction as a **supervised classification problem**.

Let the input text be:

$$
T
$$

A feature extraction function converts text into a numerical vector:

$$
x = f(T)
$$

where

$$
x \in \mathbb{R}^{25}
$$

Each feature represents a **psycholinguistic signal**.

Example:

$$
x =
[x_1, x_2, ..., x_{25}]
$$

Examples of features:

- lexical diversity
- pronoun density
- emotion word frequency
- social word frequency
- sentence length statistics

---

## Trait Prediction

Each personality trait is modeled as a binary classification problem.

For a trait \( y \in \{0,1\} \):

$$
P(y=1|x) = g(x)
$$

Where:

- \(x\) is the feature vector
- \(g\) is a trained machine learning model

In this project:

$$
g(x) = RandomForest(x)
$$

---

# Cognitive Foundations of Linguistic Personality Inference

This project is grounded in the idea that **language is a behavioral trace of cognition**.

Personality traits influence how individuals:

- attend to information
- structure thoughts
- regulate emotions
- interact socially
- communicate internal states

These differences in cognitive style manifest as **systematic patterns in language production**.

The central hypothesis of this project is therefore:

Personality → Cognitive Style → Language Patterns

By measuring linguistic signals in text, we can approximate underlying cognitive tendencies associated with personality traits.

---

## Generative Model of Language

Language is not produced by personality alone. It is influenced by several interacting factors.

A simplified model of language production can be expressed as:

$$
L = f(P, M, T) + \epsilon
$$

Where:

- \(L\) = observed language
- \(P\) = stable personality traits
- \(M\) = temporary emotional state (mood)
- \(T\) = topic of discourse
- \(\epsilon\) = noise and contextual variation

This project attempts to approximate the inverse problem:

$$
P = g(L)
$$

That is, inferring personality traits from linguistic observations.

---

## Cognitive Signals Embedded in Language

Human cognition leaves measurable traces in linguistic structure.  
The features extracted by the model attempt to capture these signals.

Examples include:

| Linguistic Signal | Cognitive Interpretation |
|------------------|--------------------------|
| lexical diversity | conceptual breadth and intellectual exploration |
| sentence length | depth of cognitive elaboration |
| pronoun usage | self-focused vs socially oriented attention |
| emotional words | affective processing tendencies |
| question frequency | curiosity and uncertainty processing |
| topic diversity | abstract thinking and conceptual exploration |

These signals act as **proxies for underlying cognitive processes**.

---

## Personality Traits as Cognitive Patterns

Each of the Big Five traits can be interpreted as a stable configuration of cognitive tendencies.

### Openness to Experience

Associated cognitive tendencies:

- curiosity
- abstract reasoning
- intellectual exploration

Expected linguistic manifestations:

- higher lexical diversity
- higher topic diversity
- more varied vocabulary

---

### Conscientiousness

Associated cognitive tendencies:

- structured thinking
- goal-directed reasoning
- disciplined cognitive organization

Expected linguistic manifestations:

- consistent sentence structure
- organized phrasing
- reduced linguistic randomness

---

### Extraversion

Associated cognitive tendencies:

- social attention
- expressive communication
- interpersonal engagement

Expected linguistic manifestations:

- higher social word frequency
- conversational language markers
- expressive punctuation

---

### Agreeableness

Associated cognitive tendencies:

- empathy
- cooperative reasoning
- social sensitivity

Expected linguistic manifestations:

- politeness markers
- supportive emotional language
- social references

---

### Neuroticism

Associated cognitive tendencies:

- emotional sensitivity
- threat detection
- anxiety-related cognitive patterns

Expected linguistic manifestations:

- higher negative emotion frequency
- anxiety-related vocabulary
- uncertainty markers

---

## Language as Behavioral Data

Traditional personality assessment relies on questionnaires.  
This project instead treats **language itself as behavioral data**.

Rather than asking individuals to self-report their personality traits, the system attempts to infer those traits from naturally produced language.

This approach aligns with emerging research areas including:

- computational psycholinguistics
- behavioral AI
- digital personality inference
- cognitive modeling of language production

Language becomes a measurable reflection of cognitive structure.

---

## Relationship to the Feature Engineering Pipeline

The 25 linguistic features used by the model are therefore not arbitrary signals.

They are designed to approximate measurable indicators of:

- cognitive complexity
- emotional processing
- social orientation
- attentional focus
- conceptual exploration

These features serve as the **observable layer connecting cognition to machine learning inference**.

# Feature Engineering

The system extracts **25 psycholinguistic features** from text.

These features approximate psychological signals used in research tools such as **LIWC (Linguistic Inquiry and Word Count)**.

---

## Structural Features

| Feature | Description |
|------|------|
| word_count | number of words |
| sentence_count | estimated number of sentences |
| lexical_diversity | unique_words / total_words |
| avg_word_length | mean word length |
| avg_sentence_length | words per sentence |

Mathematically:

$$
Lexical\ Diversity =
\frac{Unique\ Words}{Total\ Words}
$$

---

## Pronoun Usage

Pronoun usage correlates with personality and self-focus.

Features:

- first person pronouns
- plural pronouns
- second person pronouns
- third person pronouns

Pronoun density:

$$
Pronoun\ Density =
\frac{Total\ Pronouns}{Total\ Words}
$$

---

## Emotional Signals

The model measures emotional language intensity.

Emotion intensity:

$$
Emotion\ Intensity =
\frac{Positive + Negative + Anxiety + Anger}{Total\ Words}
$$

Emotion dictionaries include words such as:


love, happy, amazing
sad, angry, terrible
nervous, worried


---

## Cognitive Language

Indicators of reflective thinking:

- certainty words
- hedging words
- insight words
- causal words

Example:


think
know
because
perhaps


---

## Social Language

Indicators of social orientation:

- social words
- family words
- politeness markers

Example:


friend
family
please
thank you


---

## Punctuation Signals

Emotional intensity and conversational tone can be inferred from punctuation:


!
?


Feature examples:

- exclamation frequency
- question frequency

---

## Topic Diversity

Topic diversity estimates conceptual breadth of language.

$$
Topic\ Diversity =
\frac{Unique\ Words}{Sentence\ Count}
$$

Higher values may correlate with **Openness to Experience**.

---

# Machine Learning Model

The project uses:


RandomForestClassifier


Reasons for choosing Random Forest:

- interpretable feature importance
- robust to noisy features
- strong performance with small datasets
- minimal feature scaling required

---

## Training Process

The dataset is split into:


Training set
Validation set
Test set


Workflow:


Train → Learn parameters
Validation → Tune hyperparameters
Test → Final evaluation


---

# Model Evaluation

Evaluation metrics include:

- precision
- recall
- F1 score
- accuracy

Example output:


Training model for Extraversion

precision recall f1-score
0.49 0.49 0.49
1.61 0.61 0.61

accuracy: 0.56


Typical performance:


≈ 55% accuracy


This matches typical **psycholinguistic baseline models**.

---

# Personality Inference

After training, the models are saved:


models/
O_model.pkl
C_model.pkl
E_model.pkl
A_model.pkl
N_model.pkl


The inference script:


personality_predict.py


allows prediction from arbitrary text.

Example:


python personality_predict.py


Input:


I enjoy meeting new people and having conversations.


Output:


O: 0.46
C: 0.49
E: 0.37
A: 0.64
N: 0.49


---

# Current Limitations

The current model relies on **hand-engineered linguistic features**.

Limitations:

- no semantic understanding
- context sensitivity
- mood vs personality ambiguity

Real language can be approximated as:

$$
Language = Personality + Mood + Topic + Noise
$$

---

# Future Improvements

### Sentence Embeddings

Add semantic features using sentence transformers.

Feature vector becomes:

$$
x \in \mathbb{R}^{409}
$$


25 linguistic features
+
384 embedding features


Expected improvement:


55% → 70% accuracy


---

### Feature Importance Analysis

Use Random Forest feature importance to determine which linguistic signals predict personality traits.

Example:


Extraversion predictors:
social_words
positive_emotion
exclamation_freq


---

### Personality Explanation Engine

Provide interpretable explanations for predictions.

Example:


Prediction: High Extraversion

Driven by:

high social word usage

high positive emotion

high pronoun density


---

### Mood + Personality Modeling

Future models may separate:


stable personality traits
temporary emotional states


Mathematically:

$$
Language = Personality + Mood
$$

---

# Technologies Used

- Python
- scikit-learn
- pandas
- numpy
- joblib

---

# Research Inspiration

This project draws from research in:

- Computational Psycholinguistics
- Personality Prediction from Text
- LIWC Linguistic Analysis
- Behavioral AI

---

# License

MIT License
