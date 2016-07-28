# Neural_Conversation_Models
=================================
This implementation contains an extension of seq2seq tutorial for conversation models in Tensorflow:

1. Option to use Beam Search and Beam Size for decoding
    
2. Currently, it supports
    - Simple seq2seq  models
    - Attention based seq2seq models
    
3. To get better results use beam search during decoding / inference 


Prerequisites
-------------

- Python 2.7 or Python 3.3+
- [NLTK](http://www.nltk.org/)
- [TensorFlow](https://www.tensorflow.org/)

Data
-----
Data accepted is in the tsv format where first component is the context and second is the reply

TSV format Ubuntu Dialog Data can be found [here](https://drive.google.com/file/d/0BwPa9lrosQKdSTZxZ0tydUFGWE0/view)
 
example :-
1. What are you doing ? \t Writing seq2seq model . 

Usage
-----

To train a model with Ubuntu dataset:

    $ python neural_conversation_model.py --train_dir ubuntu/ --en_vocab_size 60000 --size 512 --data_path ubuntu/train.tsv --dev_data ubuntu/valid.tsv  --vocab_path ubuntu/60k_vocan.en --attention

To test an existing model:

    $ python neural_conversation_model.py --train_dir ubuntu/ --en_vocab_size 60000 --size 512 --data_path ubuntu/train.tsv --dev_data ubuntu/valid.tsv  --vocab_path ubuntu/60k_vocan.en --attention --decode --beam_search --beam_size 25


Todo
-----
1. Add other state of art neural models. 
2. Adding layer normalization( in progress )

https://github.com/pbhatia243/tf-layer-norm

## Contact
Parminder Bhatia, parminder@yikyakapp.com
