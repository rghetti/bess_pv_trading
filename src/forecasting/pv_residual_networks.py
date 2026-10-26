"""Market-independent neural PV corrections with explicit statistical starting points."""
from __future__ import annotations

from copy import deepcopy
import time

import numpy as np
import torch
from torch import nn

from .pv_neural_data import unpack_predictions


class WeatherResidualNetwork(nn.Module):
    """Correct baseline AC fractions using weather only; no auction identifiers.

    A zero-initialized final layer starts at the training-fitted weather regression. Sequence models see only the
    forecast curve already available at issue time, never future observations.
    """
    def __init__(self, inputs, spec):
        super().__init__()
        self.architecture = spec['architecture']
        width, depth = int(spec['width']), int(spec['depth'])
        dropout = float(spec['dropout'])
        self.stem = nn.Sequential(nn.Linear(inputs, width), nn.GELU())
        if self.architecture in {'residual_mlp', 'mixture_experts'}:
            self.body = nn.Sequential(*[layer for _ in range(depth) for layer in
                (nn.Linear(width,width), nn.GELU(), nn.Dropout(dropout))])
            outputs = int(spec['experts']) if self.architecture == 'mixture_experts' else 1
            self.head = nn.Linear(width,outputs)
            if self.architecture == 'mixture_experts':self.gate = nn.Linear(inputs,outputs)
        elif self.architecture in {'residual_tcn','direct_tcn'}:
            from .pv_neural_models import TemporalBlock
            self.body = nn.Sequential(*[TemporalBlock(width,int(spec['kernel_size']),2**i,dropout) for i in range(depth)])
            self.head = nn.Linear(width,1)
        elif self.architecture == 'bigru':
            self.body = nn.GRU(width,width,num_layers=depth,bidirectional=True,
                dropout=dropout if depth>1 else 0,batch_first=True)
            self.head = nn.Linear(2*width,1)
        elif self.architecture == 'transformer':
            layer = nn.TransformerEncoderLayer(width,int(spec['heads']),
                dim_feedforward=2*width,dropout=dropout,batch_first=True,activation='gelu')
            self.body = nn.TransformerEncoder(layer,depth,enable_nested_tensor=False)
            self.head = nn.Linear(width,1)
        else:raise ValueError('Unsupported physical residual architecture')
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,x,baseline,daylight,valid):
        """Padding never changes a retained forecast prefix or another day/view."""
        hidden = self.stem(x)
        if self.architecture == 'bigru':
            lengths = valid.sum(1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(hidden,lengths,batch_first=True,enforce_sorted=False)
            hidden,_ = self.body(packed)
            hidden,_ = nn.utils.rnn.pad_packed_sequence(hidden,batch_first=True,total_length=x.shape[1])
        elif self.architecture == 'transformer':
            hidden = self.body(hidden,src_key_padding_mask=~valid)
        elif self.architecture in {'residual_tcn','direct_tcn'}:
            hidden = self.body(hidden.transpose(1,2)).transpose(1,2)
        else:hidden = self.body(hidden)
        correction = self.head(hidden)
        if self.architecture == 'mixture_experts':
            correction = (correction*self.gate(x).softmax(-1)).sum(-1)
        else:correction = correction.squeeze(-1)
        if self.architecture == 'direct_tcn':return torch.sigmoid(correction)*daylight
        return torch.clamp(baseline+correction,0,1)*daylight


def fit_residual_network(packed,baseline,train_groups,validation_groups,spec,section,seed,rating):
    """Select epochs on the existing validation period, retaining immutable weights."""
    started=time.perf_counter();torch.set_num_threads(1)
    torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(seed)
    tensors={name:torch.as_tensor(np.array(packed[name],copy=True)) for name in ['x','y','mask','daylight']}
    tensors['valid']=torch.as_tensor(packed['indices']>=0)
    base=np.zeros(packed['indices'].shape,dtype=np.float32)
    valid=packed['indices']>=0;base[valid]=baseline[packed['indices'][valid]]/rating
    tensors['baseline']=torch.as_tensor(base)
    model=WeatherResidualNetwork(packed['x'].shape[-1],spec)
    if spec['architecture']=='direct_tcn':
        known=tensors['y'][train_groups][tensors['mask'][train_groups]]
        eps=torch.finfo(known.dtype).eps
        with torch.no_grad():model.head.bias.fill_(torch.logit(known.mean().clamp(eps,1-eps)))
    optimizer=torch.optim.AdamW(model.parameters(),lr=spec['learning_rate'],weight_decay=spec['weight_decay'])
    batch=int(section['batch_size']);history=[];best=float('inf');state=None;stale=0
    def predict(positions):
        return model(tensors['x'][positions],tensors['baseline'][positions],
            tensors['daylight'][positions],tensors['valid'][positions])
    def loss_values(prediction,target,mask,loss):
        error=prediction-target
        if loss=='huber':
            delta=section['huber_delta'];error=torch.where(error.abs()<=delta,error**2,2*delta*error.abs()-delta**2)
        else:error=error**2
        return ((error*mask).sum(1)/mask.sum(1).clamp(min=1))
    for epoch in range(1,int(section['max_epochs'])+1):
        model.train();order=rng.permutation(train_groups)
        for start in range(0,len(order),batch):
            ix=order[start:start+batch];prediction=predict(ix)
            loss=loss_values(prediction,tensors['y'][ix],tensors['mask'][ix],spec['loss']).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),section['gradient_clip_norm']);optimizer.step()
        model.eval();scores=[]
        with torch.no_grad():
            for start in range(0,len(validation_groups),batch):
                ix=validation_groups[start:start+batch]
                scores.extend(loss_values(predict(ix),tensors['y'][ix],tensors['mask'][ix],'mse').tolist())
        score=float(np.mean(scores));history.append({'epoch':epoch,'validation_nrmse':float(np.sqrt(score))})
        if score<best-section['minimum_improvement']:
            best=score;state=deepcopy(model.state_dict());stale=0
        else:stale+=1
        if stale>=section['patience']:break
    if state is None:raise ValueError('No finite neural validation checkpoint')
    model.load_state_dict(state);model.eval();pieces=[]
    with torch.no_grad():
        for start in range(0,len(base),batch):pieces.append(predict(slice(start,start+batch)).numpy())
    prediction=unpack_predictions(packed,np.concatenate(pieces),len(baseline),rating)
    return model,prediction,{'history':history,'epochs':epoch,'validation_nrmse':float(np.sqrt(best)),
        'runtime_seconds':time.perf_counter()-started,'parameters':sum(p.numel() for p in model.parameters())}
