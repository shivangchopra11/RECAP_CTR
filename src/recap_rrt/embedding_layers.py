"""Embedding layers for the ranking CTR model."""


import collections
import logging
import os


import h5py
import numpy as np
import torch
import tqdm


from . import utils




tqdm = tqdm.tqdm




class FeatureEmbeddingDict(torch.nn.Module):


 def __init__(
     self,
     feature_map,
     embedding_dim,
     embedding_initializer="partial(nn.init.normal_, std=1e-4)",
     required_feature_columns=None,
     not_required_feature_columns=None,
     use_pretrain=True,
     use_sharing=True,
 ):  # From notebook
   super(FeatureEmbeddingDict, self).__init__()
   self._feature_map = feature_map
   self.required_feature_columns = required_feature_columns
   self.not_required_feature_columns = not_required_feature_columns
   self.use_pretrain = use_pretrain
   self.embedding_initializer = utils.get_initializer(embedding_initializer)
   self.embedding_layers = torch.nn.ModuleDict()
   self.feature_encoders = torch.nn.ModuleDict()


   for (
       feature,
       feature_spec,
   ) in self._feature_map.feature_specs.items():
     if self.is_required(feature):
       feat_dim = feature_spec.get("embedding_dim", embedding_dim)
       if feature_spec.get("feature_encoder", None):
         encoder_str = feature_spec["feature_encoder"]
         try:
           if type(encoder_str) == list:
             encoder_list = []
             for enc in encoder_str:
               encoder_list.append(eval(enc))
             self.feature_encoders[feature] = torch.nn.Sequential(
                 *encoder_list
             )
           else:
             self.feature_encoders[feature] = eval(encoder_str)
         except Exception as e:
           raise ValueError(
               f"Feature encoder {encoder_str} for {feature} is not supported"
               f" or eval failed: {e}"
           )


       elif feature_spec["type"] == "embedding":
         pretrain_dim = feature_spec.get("pretrain_dim", feat_dim)
         self.feature_encoders[feature] = torch.nn.Linear(
             pretrain_dim, feat_dim, bias=False
         )


       if (
           use_sharing
           and feature_spec.get("share_embedding") in self.embedding_layers
       ):
         self.embedding_layers[feature] = self.embedding_layers[
             feature_spec["share_embedding"]
         ]
         continue


       if feature_spec["type"] == "numeric":
         self.embedding_layers[feature] = torch.nn.Linear(
             1, feat_dim, bias=False
         )
       elif feature_spec["type"] in ["categorical", "sequence"]:
         if use_pretrain and "pretrained_emb" in feature_spec:
           pass
         else:
           padding_idx = feature_spec.get("padding_idx", None)
           self.embedding_layers[feature] = torch.nn.Embedding(
               feature_spec["vocab_size"], feat_dim, padding_idx=padding_idx
           )
       elif feature_spec["type"] == "embedding":
         self.embedding_layers[feature] = torch.nn.Identity()
   self.init_weights()


 def get_feature_encoder(self, encoder_str):
   try:
     if type(encoder_str) == list:
       encoder_list = []
       for enc in encoder_str:
         encoder_list.append(eval(enc))
       return torch.nn.Sequential(*encoder_list)
     else:
       return eval(encoder_str)
   except:
     raise ValueError(
         "feature_encoder={} is not supported.".format(encoder_str)
     )


 def init_weights(self):
   for k, v in self.embedding_layers.items():
     if "share_embedding" in self._feature_map.feature_specs.get(k, {}):
       continue
     if isinstance(v, torch.nn.Embedding):
       if hasattr(self.embedding_initializer, "__call__"):
         if v.padding_idx is not None:
           # Initialize non-padding weights
           # Create a tensor for non-padding weights, initialize, then assign
           temp_weight = torch.empty_like(
               v.weight.data[v.padding_idx + 1 :]
           )  # Assuming padding_idx is the last or specific
           self.embedding_initializer(temp_weight)
           v.weight.data[v.padding_idx + 1 :] = temp_weight
           with torch.no_grad():
             v.weight.data[v.padding_idx].fill_(0)
         else:
           self.embedding_initializer(v.weight)
       else:
         logging.warning(
             "Cannot directly apply non-callable initializer in"
             " FeatureEmbeddingDict.init_weights. Relies on BaseModel init."
         )


     elif isinstance(v, torch.nn.Linear):
       if hasattr(self.embedding_initializer, "__call__"):
         self.embedding_initializer(v.weight)
       else:
         logging.warning(
             "Cannot directly apply non-callable initializer in"
             " FeatureEmbeddingDict.init_weights for Linear. Relies on"
             " BaseModel init."
         )


 def is_required(self, feature):
   feature_spec = self._feature_map.feature_specs[feature]
   if self.required_feature_columns and (
       feature not in self.required_feature_columns
   ):
     return False
   elif self.not_required_feature_columns and (
       feature in self.not_required_feature_columns
   ):
     return False
   else:
     return True


 def dict2tensor(
     self,
     embedding_dict,
     flatten_emb=False,
     feature_list=[],
     feature_source=None,
     feature_type=None,
 ):
   feature_emb_list = []
   for (
       feature,
       feature_spec,
   ) in self._feature_map.feature_specs.items():
     if not self.is_required(feature):
       continue
     if feature_list and feature not in feature_list:
       continue
     if feature_source:
       current_feature_source = feature_spec.get("source")
       if isinstance(feature_source, list):
         if current_feature_source not in feature_source:
           continue
       elif current_feature_source != feature_source:
         continue
     if feature_type:
       current_feature_type = feature_spec.get("type")
       if isinstance(feature_type, list):
         if current_feature_type not in feature_type:
           continue
       elif current_feature_type != feature_type:
         continue


     if feature in embedding_dict:
       feature_emb_list.append(embedding_dict[feature])


   if not feature_emb_list:
     logging.warning(
         "dict2tensor: No features selected or found in embedding_dict,"
         " returning empty tensor."
     )
     example_emb = (
         next(iter(embedding_dict.values())) if embedding_dict else None
     )
     device = (
         example_emb.device
         if example_emb is not None
         else self.embedding_layers[
             next(iter(self.embedding_layers))
         ].weight.device
     )


     if flatten_emb:
       return torch.empty((0, 0), device=device)  # (batch_size, 0)
     else:
       return torch.empty((0, 0, 0), device=device)  # (batch_size, 0, dim)


   if flatten_emb:
     return torch.cat(feature_emb_list, dim=-1)
   else:
     return torch.stack(feature_emb_list, dim=1)


 def forward(self, X):
   feature_emb_dict = collections.OrderedDict()
   for (
       feature,
       feature_spec,
   ) in self._feature_map.feature_specs.items():
     if self.is_required(feature):
       feature_indices = feature_spec["index"]
       if feature_spec["type"] == "numeric":
         inp = X[:, feature_indices].float().view(-1, 1)
         emb = self.embedding_layers[feature](inp)
       elif feature_spec["type"] == "categorical":
         inp = X[:, feature_indices].long()
         emb = self.embedding_layers[feature](inp)
       elif feature_spec["type"] == "sequence":
         inp = X[:, feature_indices].long()
         seq_embed_matrix = self.embedding_layers[feature](inp)
         if feature in self.feature_encoders:
           emb = self.feature_encoders[feature](seq_embed_matrix)
         else:
           if feature in self.feature_encoders:
             emb = self.feature_encoders[feature](seq_embed_matrix)
           else:
             raise ValueError(
                 f"Sequence feature {feature} has no encoder in"
                 " FeatureEmbeddingDict."
             )
       elif feature_spec["type"] == "embedding":
         inp = X[:, feature_indices].float()
         emb = self.embedding_layers[feature](inp)
         if feature in self.feature_encoders:
           emb = self.feature_encoders[feature](emb)
       else:
         continue
       feature_emb_dict[feature] = emb
   return feature_emb_dict




class EmbeddingLayer(torch.nn.Module):
 """Embedding layer for various feature types.


 This layer handles embedding of numeric, categorical, and sequence features.
 """


 def __init__(
     self,
     feature_map,
     embedding_dim,
     embedding_dropout=0,
     feature_types=["numeric", "categorical", "sequence"],
 ):
   super(EmbeddingLayer, self).__init__()
   self._feature_map = feature_map
   self._feature_types = feature_types
   self.embedding_layer = torch.nn.ModuleDict()
   self.seq_encoder_layer = torch.nn.ModuleDict()
   for feature, feature_spec in self._feature_map.feature_specs.items():
     if feature_spec["type"] == "numeric" and "numeric" in feature_types:
       self.embedding_layer.update(
           {feature: torch.nn.Linear(1, embedding_dim, bias=False)}
       )
     elif (
         feature_spec["type"] == "categorical"
         and "categorical" in feature_types
     ):
       self.embedding_layer.update({
           feature: torch.nn.Embedding(
               feature_spec["vocab_size"],
               embedding_dim,
               padding_idx=feature_spec["vocab_size"] - 1,
           )
       })
     elif feature_spec["type"] == "sequence" and "sequence" in feature_types:
       raise NotImplementedError(
           "Sequence embedding not supported for EmbeddingLayer."
       )


 def forward(self, X):
   feature_emb_list = []
   for feature, feature_spec in self._feature_map.feature_specs.items():
     if feature_spec["type"] == "numeric" and "numeric" in self._feature_types:
       inp = X[:, feature_spec["index"]].float().view(-1, 1)
       embedding_vec = self.embedding_layer[feature](inp)
     elif (
         feature_spec["type"] == "categorical"
         and "categorical" in self._feature_types
     ):
       inp = X[:, feature_spec["index"]].long()
       embedding_vec = self.embedding_layer[feature](inp)
     elif (
         feature_spec["type"] == "sequence"
         and "sequence" in self._feature_types
     ):
       inp = X[:, feature_spec["index"]].long()
       seq_embed_matrix = self.embedding_layer[feature](inp)
       embedding_vec = self.seq_encoder_layer[feature](seq_embed_matrix)
     feature_emb_list.append(embedding_vec)
   return feature_emb_list




class EmbeddingLayer_v2(torch.nn.Module):
 """Embedding layer with dropout for various feature types.


 This layer handles embedding of numeric, categorical, and sequence features
 and includes a dropout layer.
 """


 def __init__(
     self,
     feature_map,
     embedding_dim,
     embedding_dropout=0,
     feature_types=["numeric", "categorical", "sequence"],
 ):
   super(EmbeddingLayer_v2, self).__init__()
   self._feature_map = feature_map
   self._feature_types = feature_types
   self.embedding_layer = torch.nn.ModuleDict()
   self.seq_encoder_layer = torch.nn.ModuleDict()
   for feature, feature_spec in self._feature_map.feature_specs.items():
     if feature_spec["type"] == "numeric" and "numeric" in feature_types:
       self.embedding_layer.update(
           {feature: torch.nn.Linear(1, embedding_dim, bias=False)}
       )
     elif (
         feature_spec["type"] == "categorical"
         and "categorical" in feature_types
     ):
       self.embedding_layer.update({
           feature: torch.nn.Embedding(
               feature_spec["vocab_size"],
               embedding_dim,
               padding_idx=feature_spec["vocab_size"] - 1,
           )
       })
     elif feature_spec["type"] == "sequence" and "sequence" in feature_types:
       raise NotImplementedError(
           "Sequence embedding not supported for EmbeddingLayer_v2."
       )
       # self.embedding_layer.update({
       #     feature: torch.nn.Embedding(
       #         feature_spec["vocab_size"],
       #         embedding_dim,
       #         padding_idx=feature_spec["vocab_size"] - 1,
       #     )
       # })
       # if feature_spec["encoder"] != "":
       #   try:
       #     self.seq_encoder_layer.update(
       #         {feature: getattr(sequence, feature_spec["encoder"])()}
       #     )
       #   except:
       #     raise RuntimeError(
       #         "Sequence encoder={} is not supported.".format(
       #             feature_spec["encoder"]
       #         )
       #     )
       # else:
       #   self.seq_encoder_layer.update(
       #       {feature: sequence.MaskedAveragePooling()}
       #   )
   self.dropout = (
       torch.nn.Dropout2d(embedding_dropout) if embedding_dropout > 0 else None
   )


 def forward(self, X):
   feature_emb_list = []
   for feature, feature_spec in self._feature_map.feature_specs.items():
     if feature_spec["type"] == "numeric" and "numeric" in self._feature_types:
       inp = X[:, feature_spec["index"]].float().view(-1, 1)
       embedding_vec = self.embedding_layer[feature](inp)
     elif (
         feature_spec["type"] == "categorical"
         and "categorical" in self._feature_types
     ):
       inp = X[:, feature_spec["index"]].long()
       embedding_vec = self.embedding_layer[feature](inp)
     elif (
         feature_spec["type"] == "sequence"
         and "sequence" in self._feature_types
     ):
       inp = X[:, feature_spec["index"]].long()
       seq_embed_matrix = self.embedding_layer[feature](inp)
       embedding_vec = self.seq_encoder_layer[feature](seq_embed_matrix)
     feature_emb_list.append(embedding_vec)
   feature_emb = torch.stack(feature_emb_list, dim=1)
   if self.dropout is not None:
     feature_emb = self.dropout(feature_emb)
   return feature_emb




class EmbeddingLayer_v3(torch.nn.Module):
 """Embedding layer with a dictionary of embeddings.


 This layer uses an EmbeddingDictLayer to handle embeddings and includes
 a dropout layer.
 """


 def __init__(
     self,
     feature_map,
     embedding_dim,
     embedding_dropout=0,
     required_feature_columns=[],
     not_required_feature_columns=[],
 ):
   super(EmbeddingLayer_v3, self).__init__()
   self.embedding_layer = EmbeddingDictLayer(
       feature_map,
       embedding_dim,
       required_feature_columns,
       not_required_feature_columns,
   )
   self.dropout = (
       torch.nn.Dropout2d(embedding_dropout) if embedding_dropout > 0 else None
   )


 def forward(self, X):
   feature_emb_dict = self.embedding_layer(X)
   feature_emb = torch.stack(
       self.embedding_layer.dict2list(feature_emb_dict), dim=1
   )
   if self.dropout is not None:
     feature_emb = self.dropout(feature_emb)
   return feature_emb




class EmbeddingDictLayer(torch.nn.Module):
 """Embedding layer with a dictionary of embeddings.


 This layer allows for flexible embedding configurations, including pretrained
 embeddings and shared embeddings.
 """


 def __init__(
     self,
     feature_map,
     embedding_dim,
     required_feature_columns=[],
     not_required_feature_columns=[],
 ):
   super(EmbeddingDictLayer, self).__init__()
   self._feature_map = feature_map
   self.required_feature_columns = required_feature_columns
   self.not_required_feature_columns = not_required_feature_columns
   self.embedding_layer = torch.nn.ModuleDict()
   self.seq_encoder_layer = torch.nn.ModuleDict()
   for feature, feature_spec in self._feature_map.feature_specs.items():
     if self.is_required(feature):
       # Set embedding_layer according to share_embedding
       if "share_embedding" in feature_spec:
         self.embedding_layer[feature] = self.embedding_layer[
             feature_spec["share_embedding"]
         ]
       feat_emb_dim = feature_spec.get("embedding_dim", embedding_dim)
       if feature_spec["type"] == "numeric":
         if feature not in self.embedding_layer:
           self.embedding_layer[feature] = torch.nn.Linear(
               1, feat_emb_dim, bias=False
           )
       elif feature_spec["type"] == "categorical":
         if feature not in self.embedding_layer:
           padding_idx = feature_spec.get("padding_idx", None)
           embedding_matrix = torch.nn.Embedding(
               feature_spec["vocab_size"],
               feat_emb_dim,
               padding_idx=padding_idx,
           )
           if "pretrained_emb" in feature_spec:
             embeddings = self.get_pretrained_embedding(
                 feature_map.data_dir, feature, feature_spec
             )
             embedding_matrix = self.set_pretrained_embedding(
                 embedding_matrix,
                 embeddings,
                 freeze=feature_spec["freeze_emb"],
                 padding_idx=padding_idx,
             )
           self.embedding_layer[feature] = embedding_matrix
       elif feature_spec["type"] == "sequence":
         if feature not in self.embedding_layer:
           padding_idx = feature_spec["vocab_size"] - 1
           embedding_matrix = torch.nn.Embedding(
               feature_spec["vocab_size"],
               feat_emb_dim,
               padding_idx=padding_idx,
           )
           if "pretrained_emb" in feature_spec:
             embeddings = self.get_pretrained_embedding(
                 feature_map.data_dir, feature, feature_spec
             )
             embedding_matrix = self.set_pretrained_embedding(
                 embedding_matrix,
                 embeddings,
                 freeze=feature_spec["freeze_emb"],
                 padding_idx=padding_idx,
             )
           self.embedding_layer[feature] = embedding_matrix
         self.set_sequence_encoder(feature, feature_spec.get("encoder", None))


 def is_required(self, feature):
   """Check whether feature is required for embedding"""
   feature_spec = self._feature_map.feature_specs[feature]
   if len(self.required_feature_columns) > 0 and (
       feature not in self.required_feature_columns
   ):
     return False
   elif feature in self.not_required_feature_columns:
     return False
   else:
     return True


 def set_sequence_encoder(self, feature, encoder):
   raise NotImplementedError("Sequence encoder not supported.")


 def get_pretrained_embedding(self, data_dir, feature_name, feature_spec):
   pretrained_path = os.path.join(data_dir, feature_spec["pretrained_emb"])
   with h5py.File(pretrained_path, "r") as hf:
     embeddings = hf[feature_name][:]
   return embeddings


 def set_pretrained_embedding(
     self, embedding_matrix, embeddings, freeze=False, padding_idx=None
 ):
   if padding_idx is not None:
     embeddings[padding_idx] = np.zeros(embeddings.shape[-1])
   embeddings = torch.from_numpy(embeddings).float()
   embedding_matrix.weight = torch.nn.Parameter(embeddings)
   if freeze:
     embedding_matrix.weight.requires_grad = False
   return embedding_matrix


 def dict2list(self, embedding_dict):
   return list(embedding_dict.values())


 def dict2tensor(self, embedding_dict, feature_source=None, feature_type=None):
   if feature_source is not None:
     if not isinstance(feature_source, list):
       feature_source = [feature_source]
     feature_emb_list = []
     for feature, feature_spec in self._feature_map.feature_specs.items():
       if feature_spec["source"] in feature_source:
         feature_emb_list.append(embedding_dict[feature])
     return torch.stack(feature_emb_list, dim=1)
   elif feature_type is not None:
     if not isinstance(feature_type, list):
       feature_type = [feature_type]
     feature_emb_list = []
     for feature, feature_spec in self._feature_map.feature_specs.items():
       if feature_spec["type"] in feature_type:
         feature_emb_list.append(embedding_dict[feature])
     return torch.stack(feature_emb_list, dim=1)
   else:
     return torch.stack(list(embedding_dict.values()), dim=1)


 def forward(self, X):
   feature_emb_dict = collections.OrderedDict()
   for feature, feature_spec in self._feature_map.feature_specs.items():
     if feature in self.embedding_layer:
       if feature_spec["type"] == "numeric":
         inp = X[:, feature_spec["index"]].float().view(-1, 1)
         embedding_vec = self.embedding_layer[feature](inp)
       elif feature_spec["type"] == "categorical":
         inp = X[:, feature_spec["index"]].long()
         embedding_vec = self.embedding_layer[feature](inp)
       elif feature_spec["type"] == "sequence":
         inp = X[:, feature_spec["index"]].long()
         seq_embed_matrix = self.embedding_layer[feature](inp)
         if self.seq_encoder_layer[feature] is not None:
           embedding_vec = self.seq_encoder_layer[feature](seq_embed_matrix)
         else:
           embedding_vec = seq_embed_matrix
       feature_emb_dict[feature] = embedding_vec
   return feature_emb_dict




class FeatureEmbedding(torch.nn.Module):


 def __init__(
     self,
     feature_map,
     embedding_dim,
     embedding_initializer="partial(torch.nn.init.normal_, std=1e-4)",
     required_feature_columns=None,
     not_required_feature_columns=None,
     use_pretrain=True,
     use_sharing=True,
 ):
   super(FeatureEmbedding, self).__init__()
   self.embedding_layer = FeatureEmbeddingDict(
       feature_map,
       embedding_dim,
       embedding_initializer=embedding_initializer,
       required_feature_columns=required_feature_columns,
       not_required_feature_columns=not_required_feature_columns,
       use_pretrain=use_pretrain,
       use_sharing=use_sharing,
   )


 def forward(
     self, X, feature_source=None, feature_type=None, flatten_emb=False
 ):
   feature_emb_dict = self.embedding_layer(X)
   feature_emb = self.embedding_layer.dict2tensor(
       feature_emb_dict,
       flatten_emb=flatten_emb,
       feature_source=feature_source if feature_source is not None else [],
       feature_type=feature_type if feature_type is not None else [],
   )
   return feature_emb




