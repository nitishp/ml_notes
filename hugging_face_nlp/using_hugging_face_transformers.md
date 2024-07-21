# Using Hugging Face Transformers

* Library which provides a single API through which a Transformer model can be loaded, evaluated and trained

## Breaking down the 'pipeline' method
* There's 3 steps going on here:
  * Tokenizer
    * This step converts the raw text into numbers that can be passed into the model
    * Each *token* can be a word, subword or punctuation symbol
    * Example code:
    ```
    from transformers import AutoTokenizer

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    raw_inputs = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    print(inputs)
    ```
  * Model
    * Takes the output of the tokenizer and outputs a result (normally a logit that needs to be run through a softmax layer)
    * There's two versions of the AutoModel API that the library provides. `AutoModel` and `AutoModelFor...`
      * ![Transformer Model vs full model](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter2/transformer_and_head.svg)
      * In this picture `AutoModel` represents the "Transformer Model" and `AutoModelForSequenceClassification` represents the "Full Model"
      * The "Transformer Model" will output a large hidden state, whereas the "Full Model" outputs logits that can be used for the specific task (like classification)
    * Example code:
    ```
    from transformers import AutoModelForSequenceClassification

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    outputs = model(**inputs)
    print(outputs.logits.shape) # torch.Size([2, 2])
    # 2 represents batch size
    # 2 represents the possible labels
    ```
  * Post Processing
    * Takes the logits from the Model, runs a softmax layer and predicts actual classification values
    * Use `model.config.id2label` to figure out what each label means
    * Example code:
    ```
    import torch

    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)
    ```

* Diving deeper into Models
  * `AutoModel` is just a convenience wrapper for loading the model. You could just as well use something like:
  ```
  from transformers import BertModel
  model = BertModel.from_pretrained("bert-base-cased")
  ```
  * You can save models using the following code:
  ```
  model.save_pretrained("directory_on_my_computer")
  ```
* Diving deeper into Tokenizers
  * Used to convert raw text into numbers that can be fed into model
  * Some approaches to tokenizers:
    * Word based tokenizers
      * Convert each word in the sentence into a unique ID. Think something like a hashmap of word -> int
      * Cons:
        * Similar words like "dog" and "dogs" will have completely different IDs
        * The vocabulary size could be extremely large. 
        * Also need a custom token (something like "[UNK]") to represent words not in the vocabulary
    * Character based tokenizers
      * Similar in concept to word based tokenizers. But now you do it for every single character in the sentence
      * Cons:
        * Each character doesn't mean a lot on it's own. So each token doesn't represent that much
    * Subword tokenization
      * Middle ground between word-based and character-based tokenization
      * Two rules:
        * Don't decompose common words like "dog"
        * Try and decompose rare words like "annoyingly". You can split these into two tokens "annoying" and "ly"
  * After the text is tokenized, it is encoded to convert each token to a number
* Batching
  * You'll want to pad shorter input sequences to make sure all sequences in the batch have the same length
  * Make sure to also use the `attention_mask` in order to make sure the padded values aren't paid attention to in the model
  * Make sure to set the `padding=True` param in the tokenizer!