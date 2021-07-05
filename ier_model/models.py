import argparse
import torch
import torch.nn as nn

from transformers import AutoConfig
from transformers import BertModel, BertTokenizer
from transformers import RobertaModel, RobertaTokenizer
from transformers import AutoTokenizer, AutoModel


TRANSFORMER_MODELS = {
  'bert-base-uncased': (BertModel, BertTokenizer),
  'bert-large-uncased': (BertModel, BertTokenizer),
  'bert-large-uncased-whole-word-masking': (BertModel, BertTokenizer),
  'roberta-base': (RobertaModel, RobertaTokenizer),
  'roberta-large': (RobertaModel, RobertaTokenizer),
  'allenai/biomed_roberta_base': (AutoModel.from_pretrained("allenai/biomed_roberta_base"), AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")),
  'monologg/biobert_v1.1_pubmed': (AutoModel.from_pretrained("monologg/biobert_v1.1_pubmed"), AutoTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed")),
  'allenai/scibert_scivocab_uncased': (AutoModel.from_pretrained('allenai/scibert_scivocab_uncased'), AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')),
  'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext': (AutoModel.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'), AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'))
}


class SimpleDecoder(nn.Module):
  def __init__(self, output_dim, answer_num):
    super(SimpleDecoder, self).__init__()
    self.answer_num = answer_num
    self.linear = nn.Linear(output_dim, answer_num, bias=False)

  def forward(self, inputs, output_type=None):
    output_embed = self.linear(inputs)
    return output_embed


class ModelBase(nn.Module):
  def __init__(self):
    super(ModelBase, self).__init__()
    self.loss_func = nn.BCEWithLogitsLoss()
    self.sigmoid_fn = nn.Sigmoid()

  def define_loss(self, logits, targets):
    loss = self.loss_func(logits, targets)
    return loss

  def forward(self, feed_dict):
    pass


class TransformerModel(ModelBase):
  def __init__(self, args, answer_num):
    super(TransformerModel, self).__init__()
    print('In TransformerModel Initializing <{}> model...'.format(args.model_type))
    # WILL PROBABLY NEED TO FIX THIS FOR biomed_roberta
    _model_class, _tokenizer_class = TRANSFORMER_MODELS[args.model_type]
    self.transformer_tokenizer = _tokenizer_class.from_pretrained(args.model_type)
    self.transformer_config = AutoConfig.from_pretrained(args.model_type)
    #print("Config: ", self.transformer_config)  #is Roberta not getting 514?   should be this https://s3.amazonaws.com/models.huggingface.co/bert/allenai/biomed_roberta_base/config.json
    self.encoder = _model_class.from_pretrained(args.model_type)
    self.classifier = SimpleDecoder(self.transformer_config.hidden_size, answer_num)
    self.dropout = nn.Dropout(args.hidden_dropout_prob)
    self.avg_pooling = args.avg_pooling

  def forward(self, inputs, targets=None):
    #token_type_ids=inputs["token_type_ids"]
    outputs = self.encoder(
      inputs["input_ids"],
      attention_mask=inputs["attention_mask"],
      token_type_ids=inputs["token_type_ids"] if "token_type_ids" in inputs else (None)
    )
    if self.avg_pooling:  # Averaging all hidden states
      outputs = (outputs[0] * inputs["attention_mask"].unsqueeze(-1)).sum(1)\
                / inputs["attention_mask"].sum(1).unsqueeze(-1)
    else:  # Use [CLS]
      outputs = outputs[0][:, 0, :]
    outputs = self.dropout(outputs)
    logits = self.classifier(outputs)
    #print(logits.size(), logits)
    if targets is not None:
      loss = self.define_loss(logits, targets)
    else:
      loss = None
    return loss, logits


if __name__ == '__main__':

  # TEST
  parser = argparse.ArgumentParser()
  parser.add_argument("-model_type", default='bert-large-uncased-whole-word-masking')
  parser.add_argument("-hidden_dropout_prob", help="Dropout rate", default=.1, type=float)
  args = parser.parse_args()
  args.avg_pooling = False
  model = TransformerModel(args, 60000)
  for n, p in model.named_parameters():
    print(n)
  print(model.transformer_config)
