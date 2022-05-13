from sklearn.metrics import confusion_matrix
import numpy as np
import torch
import torch.nn.functional as F

def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    
    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
    """
    if use_hard_labels:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)
        # return F.cross_entropy(logits, targets, reduction=reduction) this is unstable
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss

def consistency_loss(logits_s, logits_w, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')

    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()
        select = max_probs.ge(p_cutoff).long()
        # strong_prob, strong_idx = torch.max(torch.softmax(logits_s, dim=-1), dim=-1)
        # strong_select = strong_prob.ge(p_cutoff).long()
        # select = select * strong_select * (strong_idx == max_idx)
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w / T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean(), select, max_idx.long()
    else:
        assert Exception('Not Implemented consistency_loss')
            


class Evaluator:
  def __init__(self,batch_size,n_classes,cuda_device):
    out_data = []
    self.cuda_device = cuda_device
    self.batch_size = batch_size
    self.correct = [] #Binary Array of correct/incorrect for each sample
    self.n_correct = 0 #Number of correct values
    self.n_incorrect= 0 # Number of incorrect values
    self.n_classes = n_classes
    self.n_total = 0 #Total number of values
    self.confusion_matrix = np.zeros(shape = (n_classes,n_classes)) #Matrix of prediction vs true values


  def unsup_loss(self, loader, model):
    loss = 0 
    samples = len(loader)
    for i, (x_w , x_s) in enumerate(loader):
      x = torch.cat(x_w,x_s)
      logits = model(x)
      logits_w, logits_s = logits.chunk(2)
      loss += consistency_loss(logits_s, logits_w)
    averaged_loss = loss/samples
    return averaged_loss


  def sup_loss(self, loader, model):
    loss = 0 
    samples = len(loader)
    for i, (x , y) in enumerate(loader):
      logits = model(x)
      loss += ce(logits, y)
    averaged_loss = loss/samples
    return averaged_loss

  def forward_pass(self, model , testloader, binary = False):
    if binary == True:
      s = torch.nn.Sigmoid()
      self.model_prob = np.zeros(shape = (len(testloader), 1)) # [sample , classes]
    else:
      s = torch.nn.Identity()
      self.model_prob = np.zeros(shape = (len(testloader)*self.batch_size, self.n_classes)) # [sample , classes]
    self.labels = np.zeros(shape = (len(testloader)*self.batch_size,1))
    #Pass validation set through model getting probabilities and labels
    with torch.no_grad(): #disable back prop to test the model
      num_batches = len(testloader) + self.batch_size
      for i, (inputs, labels) in enumerate( testloader ):
          start_index = i * self.batch_size
          end_index = (i * self.batch_size) + self.batch_size
          inputs = inputs.cuda(non_blocking=True, device = self.cuda_device).float()
          self.labels[start_index:end_index , :] = labels.view(self.batch_size,1).cpu().numpy()
          out = s(model(inputs)).cpu().numpy()      
          self.model_prob[start_index:end_index,:] = out

  def update_CM(self):
    self.confusion_matrix += confusion_matrix(self.labels, self.prediction,labels = list(range(self.n_classes))) 


  def predictions(self, model_is_binary = False, THRESHOLD = None):
      if model_is_binary:

        self.prediction = np.where(self.model_prob > THRESHOLD, 1,0)
        for m,p, l in zip(self.model_prob, self.prediction, self.labels):
          print("Logit: {} -- Predicted: {} label: {}".format(m,p,l))
        assert self.prediction.shape == (len(self.model_prob),1), "Shape of prediction is {} when it should be {}".format(self.prediction.shape, (len(self.model_prob),1))
      else:
        self.prediction = np.argmax(self.model_prob, axis = 1).reshape(-1,1)
        assert self.prediction.shape == (len(self.model_prob),1),  "Shape of prediction is {} when it should be {}".format(self.prediction.shape, (len(self.model_prob),1))
      self.update_CM()
      with np.printoptions(linewidth = (5*self.n_classes+20)):
        print(self.confusion_matrix)

  def TP(self, value):
    TP = self.confusion_matrix[value,value]
    return TP
  
  def Correct(self):
    return np.sum(np.diag(self.confusion_matrix))

  def TN(self, value):
    TN = self.Correct() - TP(value)
    return TN 

  def FN(self, value):
    idx = [x for x in range(self.confusion_matrix.shape[0]) if x != value]
    FN = np.sum(self.confusion_matrix[value,idx])
    return FN

  def FP(self, value):
    idx = [x for x in range(self.confusion_matrix.shape[0]) if x != value]
    FP = np.sum(self.confusion_matrix[idx,value])
    return FP

  def P(self,value):
    P = np.sum(self.confusion_matrix[value,:])
    return P

  def N(self,value):
    N = 0
    idx = [x for x in range(self.confusion_matrix.shape[0]) if x != value]
    for i in idx:
      N += np.sum(self.confusion_matrix[i,:])
    return N
  def T(self):
    return np.sum(self.confusion_matrix)
  
  def ACC(self, value):
    return ( self.TP(value) + self.TN(value) ) / ( self.P(value) + self.N(value) )

  def TPR(self, value):
    return self.TP(value)/self.P(value)

  def TNR(self, value):
    return self.TN(value)/self.N(value)

  def PPV(self, value):
    tp = self.TP(value)
    return tp/(tp + self.FP(value))

  def NPV(self, value):
    tn = self.TN(value)
    return tn/(tn + self.FN(value))

  def FNR(self, value):
    pass

  def FPR(self, value):
    pass 

  def T_ACC(self) -> float:
    return self.Correct() / self.T()
     









