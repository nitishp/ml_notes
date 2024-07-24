# Transformer Architectures For Audio

* For audio applications, it's still largely the same as for normal text. The main difference is just the transformers inputs/outputs can be audio instead of text
* Model Inputs
  * Text: Do the normal sub-word tokenization that's done for NLP style tasks
  * Audio
    * Pass raw waveform input into the model directly
    * There can be a "feature encoder" on top of the **waveform** to create a simpler embedding of the audio data
    * There can be a "feature encoder" on top of the **spectogram** to create a simpler embedding of the audio data
* Model Outputs
  * Text: Add a language modeling head to the transformer model's output
  * Spectogram: Add a spectogram head to the transformer model's output
    * You need to run this output through an ML model to actually generate a waveform that you can play

## CTC Architectures

* CTC = Connectionist Temporal Classification
* Is encoder only
* Takes the hidden states output from the encoder, feeds it into a "CTC head" to generate a series of character labels
  * It's used for automatic speech recognition
* The tricky part is actually trying to figure out to align each label with sections of the audio input
* It chunks the audio segment into 20ms chunks and predicts a character for each 20ms chunk. So what's happening under the hood is:
  * Raw audio waveform outputs 1 hidden state value in a 20ms chunk
  * Each of the hidden states are passed into a CTC layer
  * The CTC layer outputs a character
* There is some deduplication logic in the CTC. It outputs blank characters as well so it's easier to separate words
* You can also use a language model to clean up the output of a CTC

## Seq2Seq Architectures

* Seq2Seq = When we use the full encoder-decoder blocks to output sequences
* Can also be used for speech recognition (speech to text). And can also be used for text to speech
* Here the decoder block is very useful as it fills in the role of the "language model" that you can use in the CTC version of the model

## Audio Classification Architectures

* Predict a label for an audio input (either in chunks or for the whole audio file)
* One way to solve audio classification problems is to treat it like an image! 
  * You can use the audio data as a spectogram and then run classification on it!
* Any encoder-only model can be transformed into an audio classifier by adding a classification layer on top of the sequence of hidden states!