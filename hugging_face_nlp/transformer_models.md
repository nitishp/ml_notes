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
