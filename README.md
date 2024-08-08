
### Very Simple Example of Fine-tuning MiniLM 

In cases when existing embedding models don't offer the result you are looking for or the existing embeddings are too large, one approach is to fine-tune and existing LLM model. 

In this repo we fine-tune an a MiniLM model. 

By default our model return embeddings of the same same as MiniLM would (384). However, that could be changed in `params.py` the variable `OUTPUT_DIM` controls the output dimension. 

### Repo overview and how to run

`model.py` defines our model 

`train.py` does most of the work, this is the part that loads the data and trains the model. 

`train_data.csv` is example input data, with this format:

```
term_1,term_2,similarity
data science,machine learning,0.8
javascript,skateboarding,0
natural language processing,algebra,0.2
```

`eval_data.csv` has the same format. 

`params.py` specifies some of the model parameters. 

`run_model.py` is just the test of two given terms, it assumes that `similarity_model.pt` exists, which is by default created wh `train.py` runs. 

Run

```
python train.py
```

to train the model. 

`train.py` trains the model and for every epoch it evaluates it using the eval set from `eval_data.csv` if the model scores better than the previous best version, the new model is saved to `model.pt`.

The similarity score is expected to be between 0 and 1, and the loss is MSE.