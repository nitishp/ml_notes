# Building a Music Genre Classifier

* For classification tasks, you normally only really care about the encoder piece in a transformer architecture
* Types of models:
  * Keyword spotting - if an audio segment contains a keyword
  * Speech command - classification on simple command words
    * Good for transfer learning in wake words
  * Language identification - identify language being spoken
  * Zero-Shot audio classification
    * You pass in what labels could be for a given sample
    * The model gives you back a probability for each label

## Fine-tuning a model

* Make sure the sample rates for the input match the sample rate the model was trained on
* The model can expect the same feature extraction:
```
from transformers import AutoFeatureExtractor

model_id = "ntu-spml/distilhubert"
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id, do_normalize=True, return_attention_mask=True
)
feature_extractor(<sample>)
```
* You might also want to do feature scaling on the audio input
  * Set mean to 0 and variance to 1