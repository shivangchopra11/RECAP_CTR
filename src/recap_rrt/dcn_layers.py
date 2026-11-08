"""Model layers for the ranking CTR model."""


import torch


from . import base_model as bm
from . import common_layers as cl
from . import embedding_layers as el
from . import utils




class ExponentialCrossNetwork(torch.nn.Module):


 def __init__(
     self,
     input_dim,
     num_cross_layers=3,
     layer_norm=True,
     batch_norm=False,
     net_dropout=0.1,
     num_heads=1,
 ):
   super(ExponentialCrossNetwork, self).__init__()
   self.num_cross_layers = num_cross_layers
   self.layer_norm = torch.nn.ModuleList()
   self.batch_norm = torch.nn.ModuleList()
   self.dropout = torch.nn.ModuleList()
   self.w = torch.nn.ModuleList()
   self.b = torch.nn.ParameterList()
   for i in range(num_cross_layers):
     self.w.append(torch.nn.Linear(input_dim, input_dim // 2, bias=False))
     self.b.append(torch.nn.Parameter(torch.zeros((input_dim,))))
     if layer_norm:
       self.layer_norm.append(torch.nn.LayerNorm(input_dim // 2))
     if batch_norm:
       self.batch_norm.append(torch.nn.BatchNorm1d(num_heads))
     if net_dropout > 0:
       self.dropout.append(torch.nn.Dropout(net_dropout))
     torch.nn.init.uniform_(self.b[i].data)
   self.masker = torch.nn.ReLU()
   self.dfc = torch.nn.Linear(input_dim, 1)


 def forward(self, x):
   for i in range(self.num_cross_layers):
     H = self.w[i](x)
     if len(self.batch_norm) > i:
       H = self.batch_norm[i](H)
     if len(self.layer_norm) > i:
       norm_H = self.layer_norm[i](H)
       mask = self.masker(norm_H)
     else:
       mask = self.masker(H)
     H = torch.cat([H, H * mask], dim=-1)
     x = x * (H + self.b[i]) + x
     if len(self.dropout) > i:
       x = self.dropout[i](x)
   logit = self.dfc(x)
   return logit




class DenseExponentialCrossNetwork(torch.nn.Module):


 def __init__(
     self,
     input_dim,
     num_cross_layers=3,
     layer_norm=True,
     batch_norm=False,
     net_dropout=0.1,
     num_heads=1,
     use_gated_residual=False,
 ):
   super(DenseExponentialCrossNetwork, self).__init__()
   self.num_cross_layers = num_cross_layers
   self.layer_norm = torch.nn.ModuleList()
   self.batch_norm = torch.nn.ModuleList()
   self.dropout = torch.nn.ModuleList()
   self.w = torch.nn.ModuleList()
   self.b = torch.nn.ParameterList()
   for i in range(num_cross_layers):
     self.w.append(torch.nn.Linear(input_dim, input_dim // 2, bias=False))
     self.b.append(torch.nn.Parameter(torch.zeros((input_dim,))))
     if layer_norm:
       self.layer_norm.append(torch.nn.LayerNorm(input_dim // 2))
     if batch_norm:
       self.batch_norm.append(torch.nn.BatchNorm1d(num_heads))
     if net_dropout > 0:
       self.dropout.append(torch.nn.Dropout(net_dropout))
     torch.nn.init.uniform_(self.b[i].data)
   self.masker = torch.nn.ReLU()
   self.dfc = torch.nn.Linear(input_dim, 1)
   self.use_gated_residual = use_gated_residual
   if self.use_gated_residual:
     self.gates = torch.nn.ModuleList()
     for _ in range(num_cross_layers):
       self.gates.append(cl.VectorGating(input_dim))


 def forward(self, x):
   previous_activations_sum = x
   current_stream = x
   for i in range(self.num_cross_layers):
     H = self.w[i](current_stream)
     if len(self.batch_norm) > i:
       H = self.batch_norm[i](H)
     if len(self.layer_norm) > i:
       mask = self.masker(self.layer_norm[i](H))
     else:
       mask = self.masker(H)
     H = torch.cat([H, H * mask], dim=-1)
     layer_output = current_stream * (H + self.b[i])
     if self.use_gated_residual:
       current_stream = self.gates[i](previous_activations_sum, layer_output)
     else:
       current_stream = layer_output + previous_activations_sum
     if len(self.dropout) > i:
       current_stream = self.dropout[i](current_stream)
     previous_activations_sum = previous_activations_sum + current_stream
   logit = self.dfc(current_stream)
   return logit




class DenseLinearCrossNetwork(torch.nn.Module):


 def __init__(
     self,
     input_dim,
     num_cross_layers=3,
     layer_norm=True,
     batch_norm=False,
     net_dropout=0.1,
     num_heads=1,
     use_gated_residual=False,
 ):
   super(DenseLinearCrossNetwork, self).__init__()
   self.num_cross_layers = num_cross_layers
   self.layer_norm = torch.nn.ModuleList()
   self.batch_norm = torch.nn.ModuleList()
   self.dropout = torch.nn.ModuleList()
   self.w = torch.nn.ModuleList()
   self.b = torch.nn.ParameterList()
   for i in range(num_cross_layers):
     self.w.append(torch.nn.Linear(input_dim, input_dim // 2, bias=False))
     self.b.append(torch.nn.Parameter(torch.zeros((input_dim,))))
     if layer_norm:
       self.layer_norm.append(torch.nn.LayerNorm(input_dim // 2))
     if batch_norm:
       self.batch_norm.append(torch.nn.BatchNorm1d(num_heads))
     if net_dropout > 0:
       self.dropout.append(torch.nn.Dropout(net_dropout))
     torch.nn.init.uniform_(self.b[i].data)
     self.masker = torch.nn.ReLU()
     self.sfc = torch.nn.Linear(input_dim, 1)
     self.use_gated_residual = use_gated_residual
     if self.use_gated_residual:
       self.gates = torch.nn.ModuleList()
       for _ in range(num_cross_layers):
         self.gates.append(cl.VectorGating(input_dim))


 def forward(self, x):
   x0 = x
   previous_activations_sum = x
   current_stream = x
   for i in range(self.num_cross_layers):
     H = self.w[i](current_stream)
     if len(self.batch_norm) > i:
       H = self.batch_norm[i](H)
     if len(self.layer_norm) > i:
       mask = self.masker(self.layer_norm[i](H))
     else:
       mask = self.masker(H)
     H = torch.cat([H, H * mask], dim=-1)
     layer_output = x0 * (H + self.b[i])
     if self.use_gated_residual:
       current_stream = self.gates[i](previous_activations_sum, layer_output)
     else:
       current_stream = layer_output + previous_activations_sum
     if len(self.dropout) > i:
       current_stream = self.dropout[i](current_stream)
     previous_activations_sum = previous_activations_sum + current_stream
   logit = self.sfc(current_stream)
   return logit




class LinearCrossNetwork(torch.nn.Module):


 def __init__(
     self,
     input_dim,
     num_cross_layers=3,
     layer_norm=True,
     batch_norm=True,
     net_dropout=0.1,
     num_heads=1,
 ):
   super(LinearCrossNetwork, self).__init__()
   self.num_cross_layers = num_cross_layers
   self.layer_norm = torch.nn.ModuleList()
   self.batch_norm = torch.nn.ModuleList()
   self.dropout = torch.nn.ModuleList()
   self.w = torch.nn.ModuleList()
   self.b = torch.nn.ParameterList()
   for i in range(num_cross_layers):
     self.w.append(torch.nn.Linear(input_dim, input_dim // 2, bias=False))
     self.b.append(torch.nn.Parameter(torch.zeros((input_dim,))))
     if layer_norm:
       self.layer_norm.append(torch.nn.LayerNorm(input_dim // 2))
     if batch_norm:
       self.batch_norm.append(torch.nn.BatchNorm1d(num_heads))
     if net_dropout > 0:
       self.dropout.append(torch.nn.Dropout(net_dropout))
     torch.nn.init.uniform_(self.b[i].data)
   self.masker = torch.nn.ReLU()
   self.sfc = torch.nn.Linear(input_dim, 1)


 def forward(self, x):
   x0 = x
   for i in range(self.num_cross_layers):
     H = self.w[i](x)
     if len(self.batch_norm) > i:
       H = self.batch_norm[i](H)
     if len(self.layer_norm) > i:
       norm_H = self.layer_norm[i](H)
       mask = self.masker(norm_H)
     else:
       mask = self.masker(H)
     H = torch.cat([H, H * mask], dim=-1)
     x = x0 * (H + self.b[i]) + x
     if len(self.dropout) > i:
       x = self.dropout[i](x)
   logit = self.sfc(x)
   return logit


class OfficialExponentialCrossNetwork(torch.nn.Module):
  """Exact FCNv1 ECN cross operator from the official implementation."""

  def __init__(
      self,
      input_dim,
      num_cross_layers=3,
      layer_norm=False,
      batch_norm=False,
      net_dropout=0.1,
      num_heads=1,
  ):
    super().__init__()
    self.num_cross_layers = num_cross_layers
    self.layer_norm = torch.nn.ModuleList()
    self.batch_norm = torch.nn.ModuleList()
    self.dropout = torch.nn.ModuleList()
    self.w = torch.nn.ModuleList()
    self.b = torch.nn.ParameterList()
    self.gamma = torch.nn.ParameterList()
    self.beta = torch.nn.ParameterList()
    for _ in range(num_cross_layers):
      self.w.append(torch.nn.Linear(input_dim, input_dim // 2, bias=False))
      if layer_norm:
        self.layer_norm.append(torch.nn.LayerNorm(input_dim // 2))
      else:
        self.gamma.append(torch.nn.Parameter(torch.ones(input_dim // 2)))
        self.beta.append(torch.nn.Parameter(torch.zeros(input_dim // 2)))
      self.b.append(torch.nn.Parameter(torch.empty(input_dim)))
      if batch_norm:
        self.batch_norm.append(torch.nn.BatchNorm1d(num_heads))
      if net_dropout > 0:
        self.dropout.append(torch.nn.Dropout(net_dropout))
      torch.nn.init.uniform_(self.b[-1].data)
    self.dfc = torch.nn.Linear(input_dim, 1)

  def _cross_layer(self, x, x_anchor, i):
    h1 = self.w[i](x)
    if len(self.layer_norm) > i:
      h2 = self.layer_norm[i](h1)
    else:
      h2 = self.gamma[i] * h1 + self.beta[i]
    h = torch.cat([h1, h2], dim=-1)
    if len(self.batch_norm) > i:
      h = self.batch_norm[i](h)
    x = x_anchor * (h + self.b[i]) + x
    if len(self.dropout) > i:
      x = self.dropout[i](x)
    return x

  def forward(self, x):
    for i in range(self.num_cross_layers):
      x = self._cross_layer(x, x, i)
    return self.dfc(x)


class OfficialLinearCrossNetwork(torch.nn.Module):
  """Exact FCNv1 LCN cross operator from the official implementation."""

  def __init__(
      self,
      input_dim,
      num_cross_layers=3,
      layer_norm=False,
      batch_norm=False,
      net_dropout=0.1,
      num_heads=1,
  ):
    super().__init__()
    self.num_cross_layers = num_cross_layers
    self.layer_norm = torch.nn.ModuleList()
    self.batch_norm = torch.nn.ModuleList()
    self.dropout = torch.nn.ModuleList()
    self.w = torch.nn.ModuleList()
    self.b = torch.nn.ParameterList()
    self.gamma = torch.nn.ParameterList()
    self.beta = torch.nn.ParameterList()
    for _ in range(num_cross_layers):
      self.w.append(torch.nn.Linear(input_dim, input_dim // 2, bias=False))
      if layer_norm:
        self.layer_norm.append(torch.nn.LayerNorm(input_dim // 2))
      else:
        self.gamma.append(torch.nn.Parameter(torch.ones(input_dim // 2)))
        self.beta.append(torch.nn.Parameter(torch.zeros(input_dim // 2)))
      self.b.append(torch.nn.Parameter(torch.empty(input_dim)))
      if batch_norm:
        self.batch_norm.append(torch.nn.BatchNorm1d(num_heads))
      if net_dropout > 0:
        self.dropout.append(torch.nn.Dropout(net_dropout))
      torch.nn.init.uniform_(self.b[-1].data)
    self.sfc = torch.nn.Linear(input_dim, 1)

  def _cross_layer(self, x, x_anchor, i):
    h1 = self.w[i](x)
    if len(self.layer_norm) > i:
      h2 = self.layer_norm[i](h1)
    else:
      h2 = self.gamma[i] * h1 + self.beta[i]
    h = torch.cat([h1, h2], dim=-1)
    if len(self.batch_norm) > i:
      h = self.batch_norm[i](h)
    x = x_anchor * (h + self.b[i]) + x
    if len(self.dropout) > i:
      x = self.dropout[i](x)
    return x

  def forward(self, x):
    x_anchor = x
    for i in range(self.num_cross_layers):
      x = self._cross_layer(x, x_anchor, i)
    return self.sfc(x)




class DCNv3(bm.BaseModel):


 def __init__(
     self,
     feature_map,
     model_id="DCNv3",
     gpu=0,
     learning_rate=1e-3,
     embedding_dim=10,
     num_deep_cross_layers=4,
     num_shallow_cross_layers=4,
     deep_net_dropout=0.1,
     shallow_net_dropout=0.3,
     layer_norm=True,
     batch_norm=False,
     num_embedding_heads=1,
     embedding_regularizer=None,
     net_regularizer=None,
     embedding_initializer="torch.nn.init.normal_(std=1e-4)",
     recycling_embedding_factor=1,
     use_dense_ecn=True,
     use_dense_lcn=False,
     use_gated_residual=False,
     cross_style="legacy_self_mask",
     **kwargs,
 ):
   super().__init__(
       feature_map,
       model_id=model_id,
       gpu=gpu,
       embedding_regularizer=embedding_regularizer,
       net_regularizer=net_regularizer,
       embedding_initializer=embedding_initializer,  # Difference
       **kwargs,
   )
   self._embedding_initializer = embedding_initializer  # Difference


   self.embedding_layer = cl.MultiHeadFeatureEmbedding(
       feature_map, embedding_dim * num_embedding_heads, num_embedding_heads
   )
   input_dim_cross = feature_map.num_fields * (
       embedding_dim * recycling_embedding_factor
   )  # This is F * D_indiv Maybe difference?


   if cross_style not in ("legacy_self_mask", "official"):
     raise ValueError(f"Unknown cross_style={cross_style}")

   if cross_style == "official":
     if use_dense_ecn or use_dense_lcn or use_gated_residual:
       raise ValueError(
           "cross_style='official' reproduces FCNv1 and therefore requires "
           "use_dense_ecn=false, use_dense_lcn=false, use_gated_residual=false."
       )
     self.ECN = OfficialExponentialCrossNetwork(
         input_dim=input_dim_cross,
         num_cross_layers=num_deep_cross_layers,
         net_dropout=deep_net_dropout,
         layer_norm=layer_norm,
         batch_norm=batch_norm,
         num_heads=num_embedding_heads,
     )
     self.LCN = OfficialLinearCrossNetwork(
         input_dim=input_dim_cross,
         num_cross_layers=num_shallow_cross_layers,
         net_dropout=shallow_net_dropout,
         layer_norm=layer_norm,
         batch_norm=batch_norm,
         num_heads=num_embedding_heads,
     )
   else:
     # Preserve the existing tuned DCNv3 implementation unchanged.
     if use_dense_ecn:
       self.ECN = DenseExponentialCrossNetwork(
           input_dim=input_dim_cross,
           num_cross_layers=num_deep_cross_layers,
           net_dropout=deep_net_dropout,
           layer_norm=layer_norm,
           batch_norm=batch_norm,
           num_heads=num_embedding_heads,
           use_gated_residual=use_gated_residual,
       )
     else:
       self.ECN = ExponentialCrossNetwork(
           input_dim=input_dim_cross,
           num_cross_layers=num_deep_cross_layers,
           net_dropout=deep_net_dropout,
           layer_norm=layer_norm,
           batch_norm=batch_norm,
           num_heads=num_embedding_heads,
       )

     if use_dense_lcn:
       self.LCN = DenseLinearCrossNetwork(
           input_dim=input_dim_cross,
           num_cross_layers=num_shallow_cross_layers,
           net_dropout=shallow_net_dropout,
           layer_norm=layer_norm,
           batch_norm=batch_norm,
           num_heads=num_embedding_heads,
           use_gated_residual=use_gated_residual,
       )
     else:
       self.LCN = LinearCrossNetwork(
           input_dim=input_dim_cross,
           num_cross_layers=num_shallow_cross_layers,
           net_dropout=shallow_net_dropout,
           layer_norm=layer_norm,
           batch_norm=batch_norm,
           num_heads=num_embedding_heads,
       )

   self.output_activation_fn = self.get_final_activation(kwargs["task"])
   self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)


   self.init_weights(self._embedding_initializer)
   self.to_device()  # from BaseModel
   self.recycling_embedding_factor = recycling_embedding_factor


 def forward(self, inputs):
   X, y_true_original = self.inputs_to_device(inputs)


   feature_emb = self.embedding_layer(X)


   if self.recycling_embedding_factor > 1:
     feature_emb = torch.cat(
         [feature_emb] * self.recycling_embedding_factor, dim=-1
     )


   dlogit = self.ECN(feature_emb).mean(dim=1)
   slogit = self.LCN(feature_emb).mean(dim=1)
   logit = (dlogit + slogit) * 0.5


   y_pred = (
       self.output_activation_fn(logit) if self.output_activation_fn else logit
   )


   return_dict = {"y_pred": y_pred}
   # return_dict = {"y_pred": y_pred,
   #                "y_d": self.output_activation_fn(dlogit),
   #                "y_s": self.output_activation_fn(slogit)}


   if self.loss_fn is not None:
     y_d_pred = (
         self.output_activation_fn(dlogit)
         if self.output_activation_fn
         else dlogit
     )
     y_s_pred = (
         self.output_activation_fn(slogit)
         if self.output_activation_fn
         else slogit
     )


     y_true = y_true_original.view_as(y_pred)


     loss = self.loss_fn(y_pred, y_true, reduction="mean")
     loss_d = self.loss_fn(y_d_pred, y_true, reduction="mean")
     loss_s = self.loss_fn(y_s_pred, y_true, reduction="mean")


     weight_d = loss_d - loss
     weight_s = loss_s - loss
     weight_d = torch.where(weight_d > 0, weight_d, torch.zeros_like(weight_d))
     weight_s = torch.where(weight_s > 0, weight_s, torch.zeros_like(weight_s))


     # total_loss = (
     #     loss + loss_d * weight_d.detach() + loss_s * weight_s.detach()
     # )
     total_loss = loss + loss_d * weight_d + loss_s * weight_s


     reg_loss = 0
     if self._embedding_regularizer or self._net_regularizer:
       emb_reg = utils.set_regularizer(self._embedding_regularizer)
       net_reg = utils.set_regularizer(self._net_regularizer)
       for (
           name,
           param,
       ) in self.named_parameters():  # 'name' is the full parameter name
         if param.requires_grad:
           is_embedding_param = False
           if name.startswith("embedding_layer."):
             is_embedding_param = True


           if is_embedding_param:
             if self._embedding_regularizer:
               for (
                   emb_p_norm,
                   emb_lambda_val,
               ) in (
                   emb_reg
               ):  # Use emb_p_norm for the norm type (e.g., 2 for L2)
                 reg_loss += (emb_lambda_val / emb_p_norm) * torch.norm(
                     param, emb_p_norm
                 ) ** emb_p_norm
           else:
             if self._net_regularizer:
               for (
                   net_p_norm,
                   net_lambda_val,
               ) in net_reg:  # Use net_p_norm for the norm type
                 reg_loss += (net_lambda_val / net_p_norm) * torch.norm(
                     param, net_p_norm
                 ) ** net_p_norm
     return_dict["loss"] = total_loss + reg_loss
   return return_dict


# class OriginalDCNv3(BaseModel):


#   def __init__(
#       self,
#       feature_map,
#       model_id="DCNv3",
#       gpu=-1,
#       learning_rate=1e-3,
#       embedding_dim=10,
#       num_deep_cross_layers=4,
#       num_shallow_cross_layers=4,
#       deep_net_dropout=0.1,
#       shallow_net_dropout=0.3,
#       layer_norm=True,
#       batch_norm=False,
#       num_heads=1,
#       embedding_regularizer=None,
#       net_regularizer=None,
#       **kwargs,
#   ):
#     super(DCNv3, self).__init__(
#         feature_map,
#         model_id=model_id,
#         gpu=gpu,
#         embedding_regularizer=embedding_regularizer,
#         net_regularizer=net_regularizer,
#         **kwargs,
#     )
#     self.embedding_layer = MultiHeadFeatureEmbedding(
#         feature_map, embedding_dim * num_heads, num_heads
#     )
#     input_dim = feature_map.sum_emb_out_dim()
#     self.ECN = ExponentialCrossNetwork(
#         input_dim=input_dim,
#         num_cross_layers=num_deep_cross_layers,
#         net_dropout=deep_net_dropout,
#         layer_norm=layer_norm,
#         batch_norm=batch_norm,
#         num_heads=num_heads,
#     )
#     self.LCN = LinearCrossNetwork(
#         input_dim=input_dim,
#         num_cross_layers=num_shallow_cross_layers,
#         net_dropout=shallow_net_dropout,
#         layer_norm=layer_norm,
#         batch_norm=batch_norm,
#         num_heads=num_heads,
#     )
#     self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
#     self.reset_parameters()
#     self.model_to_device()


#   def forward(self, inputs):
#     X = self.get_inputs(inputs)
#     feature_emb = self.embedding_layer(X)
#     dlogit = self.ECN(feature_emb).mean(dim=1)
#     slogit = self.LCN(feature_emb).mean(dim=1)
#     logit = (dlogit + slogit) * 0.5
#     y_pred = self.output_activation(logit)
#     return_dict = {
#         "y_pred": y_pred,
#         "y_d": self.output_activation(dlogit),
#         "y_s": self.output_activation(slogit),
#     }
#     return return_dict


#   def add_loss(self, inputs):
#     return_dict = self.forward(inputs)
#     y_true = self.get_labels(inputs)
#     y_pred = return_dict["y_pred"]
#     y_d = return_dict["y_d"]
#     y_s = return_dict["y_s"]
#     loss = self.loss_fn(y_pred, y_true, reduction="mean")
#     loss_d = self.loss_fn(y_d, y_true, reduction="mean")
#     loss_s = self.loss_fn(y_s, y_true, reduction="mean")
#     weight_d = loss_d - loss
#     weight_s = loss_s - loss
#     weight_d = torch.where(
#         weight_d > 0, weight_d, torch.zeros(1).to(weight_d.device)
#     )
#     weight_s = torch.where(
#         weight_s > 0, weight_s, torch.zeros(1).to(weight_s.device)
#     )
#     loss = loss + loss_d * weight_d + loss_s * weight_s
#     return loss





class OfficialFCNv2CrossTower(torch.nn.Module):
    """Official FCNv2 hybrid cross tower for an explicit ECN/LCN schedule.

    Each depth has its own projection matrix, matching the published FCNv2
    baseline. ``schedule[i]`` is either ``"ecn"`` (current-state anchor) or
    ``"lcn"`` (original-input anchor).
    """

    def __init__(
        self,
        input_dim,
        schedule,
        layer_norm=False,
        batch_norm=False,
        net_dropout=0.1,
        num_heads=1,
        norm_cap=None,
    ):
        super().__init__()
        # norm_cap: optional per-layer RMS cap on the state (the recursive tower's cross_norm_cap rule).
        # None keeps the official FCNv2 forward exactly.  Needed for towers with long runs of
        # exponential layers, whose state overflows fp32 during training (order 2^k).
        self.norm_cap = float(norm_cap) if norm_cap else None
        self.schedule = tuple(str(mode).lower() for mode in schedule)
        if not self.schedule or any(mode not in ("ecn", "lcn") for mode in self.schedule):
            raise ValueError("schedule must be a non-empty sequence containing only 'ecn'/'lcn'")

        hidden_dim = input_dim // 2
        self.w = torch.nn.ModuleList()
        self.layer_norm = torch.nn.ModuleList()
        self.batch_norm = torch.nn.ModuleList()
        self.dropout = torch.nn.ModuleList()
        self.b = torch.nn.ParameterList()
        self.gamma = torch.nn.ParameterList()
        self.beta = torch.nn.ParameterList()

        for _ in self.schedule:
            self.w.append(torch.nn.Linear(input_dim, hidden_dim, bias=False))
            if layer_norm:
                self.layer_norm.append(torch.nn.LayerNorm(hidden_dim))
            else:
                self.gamma.append(torch.nn.Parameter(torch.ones(hidden_dim)))
                self.beta.append(torch.nn.Parameter(torch.zeros(hidden_dim)))
            self.b.append(torch.nn.Parameter(torch.empty(input_dim)))
            if batch_norm:
                self.batch_norm.append(torch.nn.BatchNorm1d(num_heads))
            if net_dropout > 0:
                self.dropout.append(torch.nn.Dropout(net_dropout))
            torch.nn.init.uniform_(self.b[-1].data)

        self.fc = torch.nn.Linear(input_dim, 1)

    def _cross_layer(self, x, x_anchor, i):
        h1 = self.w[i](x)
        if len(self.layer_norm) > i:
            h2 = self.layer_norm[i](h1)
        else:
            h2 = self.gamma[i] * h1 + self.beta[i]
        h = torch.cat([h1, h2], dim=-1)
        if len(self.batch_norm) > i:
            h = self.batch_norm[i](h)
        x = x_anchor * (h + self.b[i]) + x
        if self.norm_cap is not None:
            rms = x.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
            x = x * (self.norm_cap / rms).clamp(max=1.0)
        if len(self.dropout) > i:
            x = self.dropout[i](x)
        return x

    def forward(self, x):
        x0 = x
        for i, mode in enumerate(self.schedule):
            anchor = x if mode == "ecn" else x0
            x = self._cross_layer(x, anchor, i)
        return self.fc(x)


class FCNv2(bm.BaseModel):
    """Exact two-tower FCNv2 baseline used by the official Criteo run.

    Tower 1: ECN x ``exp_num_layers`` then LCN x ``lin_num_layers``.
    Tower 2: LCN x ``lin_num_layers`` then ECN x ``exp_num_layers``.
    The two logits are averaged before the output activation and optimized with
    the official Tri-BCE objective.
    """

    def __init__(
        self,
        feature_map,
        model_id="FCNv2",
        gpu=0,
        learning_rate=1e-3,
        embedding_dim=16,
        exp_num_layers=5,
        lin_num_layers=5,
        net_dropout=0.1,
        batch_norm=False,
        layer_norm=False,
        num_heads=1,
        embedding_regularizer=None,
        net_regularizer=None,
        embedding_initializer="torch.nn.init.normal_(std=1e-4)",
        **kwargs,
    ):
        super().__init__(
            feature_map,
            model_id=model_id,
            gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs,
        )
        self._embedding_initializer = embedding_initializer
        self.embedding_layer = cl.MultiHeadFeatureEmbedding(
            feature_map, embedding_dim * num_heads, num_heads
        )
        input_dim = feature_map.num_fields * embedding_dim

        self.E2LCN = OfficialFCNv2CrossTower(
            input_dim=input_dim,
            schedule=["ecn"] * int(exp_num_layers) + ["lcn"] * int(lin_num_layers),
            net_dropout=net_dropout,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
            num_heads=num_heads,
        )
        self.L2ECN = OfficialFCNv2CrossTower(
            input_dim=input_dim,
            schedule=["lcn"] * int(lin_num_layers) + ["ecn"] * int(exp_num_layers),
            net_dropout=net_dropout,
            batch_norm=batch_norm,
            layer_norm=layer_norm,
            num_heads=num_heads,
        )

        self.output_activation_fn = self.get_final_activation(kwargs.get("task", "binary_classification"))
        # Official FCNv2/FuxiCTR-2.0.1 resets Linear layers on CPU *after* the
        # embedding tables have initialized themselves, then moves the model to
        # GPU.  Do not call init_weights() here: that would initialize embeddings
        # a second time and also shift the seed-2025 RNG trajectory.
        self.reset_parameters()
        self.compile(kwargs["optimizer"], kwargs["loss"], learning_rate)
        self.to_device()

    def _regularization_loss(self):
        reg_loss = 0.0
        if not (self._embedding_regularizer or self._net_regularizer):
            return reg_loss
        emb_reg = utils.set_regularizer(self._embedding_regularizer)
        net_reg = utils.set_regularizer(self._net_regularizer)
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            pairs = emb_reg if name.startswith("embedding_layer.") else net_reg
            for p_norm, weight in pairs:
                reg_loss = reg_loss + (weight / p_norm) * torch.norm(param, p_norm) ** p_norm
        return reg_loss

    def forward(self, inputs):
        X, y_true_original = self.inputs_to_device(inputs)
        feature_emb = self.embedding_layer(X)

        e2l_logit = self.E2LCN(feature_emb).mean(dim=1)
        l2e_logit = self.L2ECN(feature_emb).mean(dim=1)
        logit = 0.5 * (e2l_logit + l2e_logit)

        y_pred = self.output_activation_fn(logit) if self.output_activation_fn else logit
        return_dict = {"y_pred": y_pred}

        if self.loss_fn is not None:
            y_true = y_true_original.view_as(y_pred)
            e2l_pred = self.output_activation_fn(e2l_logit) if self.output_activation_fn else e2l_logit
            l2e_pred = self.output_activation_fn(l2e_logit) if self.output_activation_fn else l2e_logit

            loss = self.loss_fn(y_pred, y_true, reduction="mean")
            loss_e2l = self.loss_fn(e2l_pred, y_true, reduction="mean")
            loss_l2e = self.loss_fn(l2e_pred, y_true, reduction="mean")
            weight_e2l = torch.clamp(loss_e2l - loss, min=0.0)
            weight_l2e = torch.clamp(loss_l2e - loss, min=0.0)
            total_loss = loss + loss_e2l * weight_e2l + loss_l2e * weight_l2e
            return_dict["loss"] = total_loss + self._regularization_loss()

        return return_dict
