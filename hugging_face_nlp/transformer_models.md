# Transformer Models

* What is NLP? 
  * Field of understanding everything related to the human language
  * Not just text, can also be audio and analyzing text in images
* Pipeline 
  * Highest level abstraction of the HuggingFace infrastructure
  * Takes in a string input, does pre-processing, model prediction and post processing to give a nice answer  
  * Can plug in with different models from Model Hub
  * Example:
  ```
  from transformers import pipeline

  zero_shot = pipeline('zero-shot-classification', model="X")
  zero_shot("Hi this is a text", labels=["OK", "Text"])
  ```
* Hugging Face Model Hub
  * Contains other models you can hook up with the pipeline

## Transformer Details
* A transformer is a model that extracts statistical data from a language
  * It's unsupervised
  * It doesn't do anything on it's own. It is used as part of step before doing transfer learning
  * They're really big and can take a very long time to train
* General components
  * Types of models
    * Encoder-only: Takes a bunch of input and "understands" it. 
      * Often called "Auto-encoding" models.
      * BERT-like models
      * Takes in a sentence of words, outputs a `vector<float>` for each word (feature vector)
        * The `vector<float>` for each word are affected by other words in the sentence. On both sides
      * Used for things like classification
    * Decoder: Used for generating new outputs from the input. Think text generation
      * Often called "auto-regressive" models
      * GPT-like models
      * Takes in a sentence of words, outputs a `vector<float>` for each word
        * The `vector<float>` for each word are affected by other words in the sentence. Only on the left side!
        * This is called "masked attention"
      * Used for generating sequence of words (next word prediction)
      * "Auto regressive" because it takes the model's output and uses it as the next input
    * Encoder-Decoder: Combo of the two. 
      * Often called "Sequence-to-sequence" models
      * BART/T5-like models
      * Combo of the two models above
        * The weights between the two of these components are completely independent
      * Used for things like text summarization or translation
  * Attention layers
    * Used as a component to decide which sections of a sentence to pay attention to.
* Original model
  ![Original Transformer model](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/transformers.svg)
* 