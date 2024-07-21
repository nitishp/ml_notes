# Fine-Tuning a Pretrained Model

* Processing the data
  * You can load datasets from Hugging Face
  * You can simply use the `load_datasets` function to load any dataset
  * Example code:
  ```
  ## Returns
  # Input IDs
  # Attention Mask
  # token_type_ids : used to distinguish which tokens belong to which sentence
  def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
  
  raw_datasets = load_dataset("glue", "mrpc")
  tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
  ```
  * Collator: Function responsible for putting items into a batch
    * You normally do any padding needed inside of this function
    * Example code:
    ```
    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    batch = data_collator(samples)
    ```
    * Recall that the Collate function is used in the `DataLoader`. Something like this:
    ```
    train_loader = data.DataLoader(dataset=train_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        collate_fn=collate_fn,
                                        **kwargs)
    ```

* Fine-tuning the model
  * Use built in Trainer API
    * You can use the following snippet to train the model
    * Example code:
      ```
      import evaluate
      from transformers import Trainer

      def compute_metrics(eval_preds):
        metric = evaluate.load("glue", "mrpc")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

      trainer = Trainer(
          model,
          training_args,
          train_dataset=tokenized_datasets["train"],
          eval_dataset=tokenized_datasets["validation"],
          data_collator=data_collator,
          tokenizer=tokenizer,
          compute_metrics=compute_metrics,
      )
      trainer.train()
      ```
  * Use a PyTorch evaluation loop
    * Use DataLoaders. Like so:
      ```
      train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
      )
      ```
    * The training loop will look like so:
      ```
      from transformers import AdamW

      optimizer = AdamW(model.parameters(), lr=5e-5)
      model.train()
      for epoch in range(num_epochs):
        for batch in train_dataloader:
          batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs.loss
          loss.backward()

          optimizer.step()
          lr_scheduler.step()
          # Don't forget to zero this out so it doesn't apply to future iterations of the loop
          optimizer.zero_grad()
      ```
