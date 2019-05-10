# NLP-Projects
Some fun NLP projects based on Pytorch.
- Pytorch: 1.0.1
- Python: 3.6
## 1. LSTM part-of-speech tagger with character-level features
### Train and test

- The following command starts training.

```
python lstm_tag_plus.py
```
### Refs
- [Exercise: Augmenting the LSTM part-of-speech tagger with character-level features](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py)
- [pytorch-discuss](https://discuss.pytorch.org/t/implementation-augmenting-the-lstm-part-of-speech-tagger-with-character-level-features/10221/5)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## 2. 普通话-四川话翻译器
### Train and test

- The following command starts training.

```
python main.py
```
- Examples:
  * 普通话> 你想做什么  四川话: 你想做啥子
  * 普通话> 你怎么这么那个  四川话: 你啷个这么那个
  * 普通话> 我明天不上班 四川话: 老子明天不上班
- A jupyter-notebook version can be found in pt-sc/pt_sc.ipynb.

### Refs
- [pytorch-official-tutorials](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- [jieba-introduction](https://github.com/fxsjy/jieba)

