# Working with Audio Data

## Intro to Audio Data
* The main difference between .wav, .mp3 is just the compression of the digital info for an audio
* Sampling
  * Refers to how many times in a given second audio samples are recorded
  * Often a 16k sampling rate is good enough for most ML models
  * All examples in the training / validation set should have the same sample rate. 
    * ML models treat audio data as a sequence of values
    * You want to make sure similar audio clips have the same sequence of values!
  * If sample rates don't match. You can always "Resample" as part of audio preprocessing
* Amplitude
  * The value recorded at each sample is the amplitude of the wave. 
  * The bit depth controls the precision of this amplitude
  * 16 bit is normally good enough for models
  * If it returns a floating point number, it's normally between [-1.0, 1.0]
* Visualizing Data
  * You can use `librosa` for plotting
  * You can visualize the data in the following domains:
    * Time
    * Frequency (using FFT)
    * Spectogram
      * Used to track frequency changes over time. 
      * Sort of a combo of the time and frequency domains
      * Takes multiple FFTs each over a small timestamp, and then stacks them together
    * Mel Spectogram
      * Similar to Spectogram, but a slight tweak on the y-axis
        * It still measures frequencies but not on a linear scale
      * Also filters out some uninteresting frequencies
      * Commonly used for ML tasks

## Preprocessing an Audio Dataset
* Common pre-processing actions:
  * Resampling Data
    * This is a common step needed to make sure that the training/inference data is in the same sampling rate that the model expects
    * In Hugging Face:
      ```
      from datasets import Audio

      minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
      ```
  * Filtering dataset for any random qualities like duration
  * Pre-processing data to match the model. Examples include:
    * Padding/truncating audio data so that it's 30 seconds long
    * Convert data to mel spectograms
