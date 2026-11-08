import torch
import tqdm


from . import embedding_layers as el
from . import utils




tqdm = tqdm.tqdm




class DNN_Layer(torch.nn.Module):
 """Deep Neural Network Layer.


 This layer constructs a multi-layered perceptron with specified hidden units,
 activations, dropout, and batch normalization.
 """


 def __init__(
     self,
     input_dim,
     output_dim=None,
     hidden_units=[],
     hidden_activations="ReLU",
     final_activation=None,
     dropout_rates=[],
     batch_norm=False,
     use_bias=True,
 ):
   super(DNN_Layer, self).__init__()
   dense_layers = []
   if not isinstance(dropout_rates, list):
     dropout_rates = [dropout_rates] * len(hidden_units)
   if not isinstance(hidden_activations, list):
     hidden_activations = [hidden_activations] * len(hidden_units)
   hidden_activations = [utils.set_activation(x) for x in hidden_activations]
   hidden_units = [input_dim] + hidden_units
   for idx in range(len(hidden_units) - 1):
     dense_layers.append(
         torch.nn.Linear(
             hidden_units[idx], hidden_units[idx + 1], bias=use_bias
         )
     )
     if batch_norm:
       dense_layers.append(torch.nn.BatchNorm1d(hidden_units[idx + 1]))
     if hidden_activations[idx]:
       dense_layers.append(hidden_activations[idx])
     if dropout_rates[idx] > 0:
       dense_layers.append(torch.nn.Dropout(p=dropout_rates[idx]))
   if output_dim is not None:
     dense_layers.append(
         torch.nn.Linear(hidden_units[-1], output_dim, bias=use_bias)
     )
   if final_activation is not None:
     dense_layers.append(utils.set_activation(final_activation))
   self.dnn = torch.nn.Sequential(*dense_layers)  # * used to unpack list


 def forward(self, inputs):
   return self.dnn(inputs)




class LR_Layer(torch.nn.Module):
 """Linear Regression Layer.


 This layer performs a linear combination of input features, optionally
 including a bias term and a final activation.
 """


 def __init__(self, feature_map, final_activation=None, use_bias=True):
   super(LR_Layer, self).__init__()
   self.bias = (
       torch.nn.Parameter(torch.zeros(1), requires_grad=True)
       if use_bias
       else None
   )
   self.final_activation = final_activation
   # A trick for quick one-hot encoding in LR
   self.embedding_layer = el.EmbeddingLayer(feature_map, 1)


 def forward(self, X):
   embed_weights = self.embedding_layer(X)
   output = torch.stack(embed_weights).sum(dim=0)
   if self.bias is not None:
     output += self.bias
   if self.final_activation is not None:
     output = self.final_activation(output)
   return output




class FeatureSelection(torch.nn.Module):


 def __init__(
     self,
     feature_map,
     feature_dim,
     embedding_dim,
     fs_hidden_units=[],
     fs1_context=[],
     fs2_context=[],
 ):
   super(FeatureSelection, self).__init__()
   self.fs1_context = fs1_context
   if len(fs1_context) == 0:
     self.fs1_ctx_bias = torch.nn.Parameter(torch.zeros(1, embedding_dim))
   else:
     self.fs1_ctx_emb = el.FeatureEmbedding(
         feature_map, embedding_dim, required_feature_columns=fs1_context
     )
   self.fs2_context = fs2_context
   if len(fs2_context) == 0:
     self.fs2_ctx_bias = torch.nn.Parameter(torch.zeros(1, embedding_dim))
   else:
     self.fs2_ctx_emb = el.FeatureEmbedding(
         feature_map, embedding_dim, required_feature_columns=fs2_context
     )
   self.fs1_gate = DNN_Layer(
       input_dim=embedding_dim * max(1, len(fs1_context)),
       output_dim=feature_dim,
       hidden_units=fs_hidden_units,
       hidden_activations="ReLU",
       final_activation="sigmoid",
       batch_norm=False,
       dropout_rates=0.0,
   )
   self.fs2_gate = DNN_Layer(
       input_dim=embedding_dim * max(1, len(fs2_context)),
       output_dim=feature_dim,
       hidden_units=fs_hidden_units,
       hidden_activations="ReLU",
       final_activation="sigmoid",
       batch_norm=False,
       dropout_rates=0.0,
   )


 def forward(self, X, flat_emb):
   if len(self.fs1_context) == 0:
     fs1_input = self.fs1_ctx_bias.repeat(flat_emb.size(0), 1)
   else:
     fs1_input = self.fs1_ctx_emb(X).flatten(start_dim=1)
   gt1 = self.fs1_gate(fs1_input) * 2
   feature1 = flat_emb * gt1
   if len(self.fs2_context) == 0:
     fs2_input = self.fs2_ctx_bias.repeat(flat_emb.size(0), 1)
   else:
     fs2_input = self.fs2_ctx_emb(X).flatten(start_dim=1)
   gt2 = self.fs2_gate(fs2_input) * 2
   feature2 = flat_emb * gt2
   return feature1, feature2




class MultiHeadFeatureEmbedding(torch.nn.Module):


 def __init__(self, feature_map, embedding_dim, num_heads=2):
   super(MultiHeadFeatureEmbedding, self).__init__()
   self.num_heads = num_heads
   self.embedding_layer = el.FeatureEmbedding(feature_map, embedding_dim)


 def forward(self, X):  # H = num_heads
   feature_emb = self.embedding_layer(X)  # B × F × D
   multihead_feature_emb = torch.tensor_split(
       feature_emb, self.num_heads, dim=-1
   )
   multihead_feature_emb = torch.stack(
       multihead_feature_emb, dim=1
   )  # B × H × F × D/H
   multihead_feature_emb1, multihead_feature_emb2 = torch.tensor_split(
       multihead_feature_emb, 2, dim=-1
   )  # B × H × F × D/2H
   (
       multihead_feature_emb1,
       multihead_feature_emb2,
   ) = multihead_feature_emb1.flatten(
       start_dim=2
   ), multihead_feature_emb2.flatten(
       start_dim=2
   )  # B × H × FD/2H; B × H × FD/2H
   multihead_feature_emb = torch.cat(
       [multihead_feature_emb1, multihead_feature_emb2], dim=-1
   )
   return multihead_feature_emb  # B × H × FD/H




class VectorGating(torch.nn.Module):


 def __init__(self, input_dim, input_dependent=False):
   super(VectorGating, self).__init__()
   self.gates = torch.nn.Parameter(torch.full((input_dim,), -3.0))
   self.input_dependent = input_dependent


 def forward(self, x, y):
   gates_input = self.gates * x if self.input_dependent else self.gates
   postfactor = torch.sigmoid(gates_input)
   prefactor = torch.sqrt((1 - postfactor) * (1 + postfactor))
   return prefactor * x + postfactor * y




