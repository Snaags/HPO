from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt 
import numpy as np
from HPO.algorithms.NASWOT import naswot
import torch
import torch.nn.functional as F
from HPO.utils.time_series_augmentation_torch import  jitter, scaling, window_warp,crop, cutout
from HPO.utils.train_utils import collate_fn_padd, collate_fn_padd_x
import random
import copy
from torch.utils.data import Sampler
from sklearn.model_selection import StratifiedKFold
import pandas as pd

class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        try:
            from sklearn.model_selection import StratifiedShuffleSplit
        except:
            print('Need scikit-learn for this functionality')
        import numpy as np
        
        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = th.randn(self.class_vector.size(0),2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)

class StratifiedBatchSampler:
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.cpu().numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        print("Number of samples: {} -- Batch_Size: {}".format(len(y),batch_size))
        n_batches = int(len(y) / batch_size)
        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y)

def augment(x, y = None, settings = None ,device = None):
  augs = [jitter, scaling, window_warp,crop]
  
  rate = 0.5
  if device == None:
    for func in augs:
      if random.random() < rate:
        x,y = func(x,y)
  else:
    if settings != None:
      rates = settings["rates"]
      hps = settings["hps"]
      for rate, hp,func in zip(rates, hps,augs):
        if random.random() < rate:
          x,y = func(x,y,hp, device = device)
    else:
      for func in augs:
        if random.random() < rate:
          x,y = func(x,y, device = device)
  return x  
def ce_loss(logits, targets, use_hard_labels=True, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    Args:
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
        return masked_loss.mean()
    else:
        assert Exception('Not Implemented consistency_loss')
            


class Evaluator:
  def __init__(self,batch_size,n_classes,cuda_device, testloader = None):
    out_data = []
    self.cuda_device = cuda_device
    self.testloader = testloader
    self.batch_size = batch_size
    self.correct = [] #Binary Array of correct/incorrect for each sample
    self.n_correct = 0 #Number of correct values
    self.n_incorrect= 0 # Number of incorrect values
    self.n_classes = n_classes
    self.n_total = 0 #Total number of values
    self.confusion_matrix = np.zeros(shape = (n_classes,n_classes)) #Matrix of prediction vs true values
  
  def score_naswot(self,model,loader):
    nw = naswot(model,loader,self.batch_size,self.cuda_device)
    return nw.score()
  
  def unsup_loss(self, model,loader,binary = False, n_augment = 1):
    loss = 0 
    S = torch.nn.Sigmoid()
    if binary == True:
      lossFn = F.binary_cross_entropy_with_logits
    else:
      lossFn = F.cross_entropy
    samples = len(loader)
    with torch.no_grad(): #disable back prop to test the model
      for i, (x,_) in enumerate(loader):
        x = x.cuda(non_blocking=True, device = self.cuda_device).float()
        for a in range(n_augment):
          x_1 = augment(x,device = self.cuda_device)
          x_2 = augment(x,device = self.cuda_device)
          logits_1 = model(x_1)
          logits_2 = model(x_2)
          loss += lossFn(S(logits_1), S(logits_2))
      averaged_loss = loss/(samples*n_augment)
    return averaged_loss

  def loss_over_sample_size(self,model,dataset):
      n = 2
      u_loss = {}
      u10_loss = {}
      u100_loss = {}
      s_loss = {}
      while n < 16:
        s, u, u10,u100 = self.loss_distribution(model,dataset , sample_per_class = n)
        u_loss[n] = copy.deepcopy(u)
        u10_loss[n] = copy.deepcopy(u10)
        u100_loss[n] = copy.deepcopy(u100)
        s_loss[n] = copy.deepcopy(s)
        n = n*2
      df_supervised = pd.DataFrame(s_loss.items(),columns = ["num samples", "supervised scores"])
      df_supervised = df_supervised.explode("supervised scores")
      df_unsupervised = pd.DataFrame(u_loss.items(),columns = ["num samples", "unsupervised scores"])
      df_unsupervised = df_unsupervised.explode("unsupervised scores")
      df_unsupervised10 = pd.DataFrame(u10_loss.items(),columns = ["num samples", "unsupervised scores"])
      df_unsupervised = df_unsupervised.explode("unsupervised scores")
      df_unsupervised100 = pd.DataFrame(u100_loss.items(),columns = ["num samples", "unsupervised scores"])
      df_unsupervised = df_unsupervised.explode("unsupervised scores")
      df_loss = pd.DataFrame()
      df_loss["samples"] = df_supervised["num samples"]
      df_loss["supervised"] = df_supervised["supervised scores"]
      df_loss["unsupervised"] = df_unsupervised["unsupervised scores"]
      df_loss["unsupervised10"] = df_unsupervised10["unsupervised scores"]
      df_loss["unsupervised100"] = df_unsupervised100["unsupervised scores"]
      return df_loss
  def loss_distribution(self, model, dataset, loss = "all", sample_per_class = 50):
      labels = dataset.get_labels()
      MAX_SIZE = 500
      print(labels)
      
      strat_sampler = StratifiedBatchSampler(y = labels , batch_size = sample_per_class*21)

      testloader = torch.utils.data.DataLoader(
                          dataset,collate_fn = collate_fn_padd,batch_sampler = strat_sampler)
      sup_losses = []
      usup_losses = []
      usup10_losses = []
      usup100_losses = []
      for x,y in testloader:
        chunks = -(len(y)//-MAX_SIZE)
        x_i = x.chunk(chunks)
        y_i = y.chunk(chunks)
        subsample = [(x_j,y_j) for x_j, y_j in zip(x_i,y_i)]
           
        s_loss = self.sup_loss(model,subsample)
        u_loss = self.unsup_loss(model,subsample)
        u10_loss = self.unsup_loss(model,subsample, n_augment = 10)
        u100_loss = self.unsup_loss(model,subsample, n_augment = 100)
        sup_losses.append(s_loss.item())
        usup_losses.append(u_loss.item())
        usup10_losses.append(u10_loss.item())
        usup100_losses.append(u100_loss.item())
        
      return sup_losses, usup_losses , usup10_losses, usup100_losses

  def sup_loss(self, model,loader,n_augment = 1 , binary = False):
    loss = 0 
    S = torch.nn.Sigmoid()
    samples = len(loader)
    if binary == True:
      lossFn = F.binary_cross_entropy_with_logits
    else:
      lossFn = F.cross_entropy
    with torch.no_grad(): #disable back prop to test the model
      for i, (x , y) in enumerate(loader):
        for i in range(n_augment):
          x = x.cuda(non_blocking=True, device = self.cuda_device).float()
          y = y.cuda(non_blocking=True, device = self.cuda_device).long()
          x = augment(x,device = self.cuda_device)
          logits = model(x)
          loss += lossFn(S(logits), y)
      averaged_loss = loss/samples
    return averaged_loss

  def c_loss(self,model,loader,n_augment = 1,binary = False):
    loss = 0 
    s_rates = [0.7, 0.7, 0.9,0.7]
    s_hps = [0.15, 0.12, 4,0.49]
    w_rates = [0.3, 0.3, 0.5,0.3]
    w_hps = [0.015, 0.012, 2,0.1]
    w_settings = {"rates": w_rates,"hps":w_hps}
    s_settings = {"rates": s_rates,"hps":s_hps}
    S = torch.nn.Sigmoid()
    if binary == True:
      lossFn = F.binary_cross_entropy_with_logits
    else:
      lossFn = F.cross_entropy
    samples = len(loader)
    with torch.no_grad(): #disable back prop to test the model
      for i, (x,_) in enumerate(loader):
        x = x.cuda(non_blocking=True, device = self.cuda_device).float()
        for a in range(n_augment):
          x_s = augment(x,s_settings,device = self.cuda_device)
          x_w = augment(x,w_settings,device = self.cuda_device)
          logits_w = model(x_w)
          logits_s = model(x_s)
          loss += consistency_loss(logits_s, logits_w)
      averaged_loss = loss/(samples*n_augment)
    return averaged_loss
  def ROC(self,fold = 0):
    fpr , tpr, thresholds = roc_curve(self.labels, self.model_prob) 
    roc_auc = auc(fpr,tpr)
    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.savefig("ROC_{}".format(fold))
    return roc_auc
  def forward_pass(self, model , testloader = None, binary = False, subset = None,n_iter = 1):
    if testloader != None:
      self.testloader = testloader
    elif self.testloader == None and testloader == None:
      raise Exception("No dataloader passed at object initialisation or at forward pass")

    if binary == True:
      s = torch.nn.Sigmoid() #torch.nn.Identity()# torch.nn.Sigmoid()
      self.model_prob = np.zeros(shape = (len(self.testloader)*self.batch_size*n_iter, 1)) # [sample , classes]
    else:
      s = torch.nn.Identity()
      self.model_prob = np.zeros(shape = (len(self.testloader)*self.batch_size*n_iter, self.n_classes)) # [sample , classes]
    self.labels = np.zeros(shape = (len(self.testloader)*self.batch_size*n_iter,1))

    #Pass validation set through model getting probabilities and labels
    with torch.no_grad(): #disable back prop to test the model
      for n in range(n_iter):
        for i, (inputs, labels) in enumerate( self.testloader ):
            start_index = i * self.batch_size + (n * len(self.testloader) * self.batch_size)
            end_index = (i * self.batch_size + (n * len(self.testloader) * self.batch_size)) + self.batch_size
            inputs = inputs.cuda(non_blocking=True, device = self.cuda_device).float()
            self.labels[start_index:end_index , :] = labels.view(self.batch_size,1).cpu().numpy()
            out = s(model(inputs)).cpu().numpy()
            self.model_prob[start_index:end_index,:] = out
            if subset != None:
              if i > subset:
                self.labels = self.labels[:end_index,:]
                self.model_prob = self.model_prob[:end_index, :]
                break
  def update_CM(self):
    self.confusion_matrix += confusion_matrix(self.labels, self.prediction,labels = list(range(self.n_classes))) 
  def reset_cm(self):
    self.confusion_matrix = np.zeros(shape = (self.n_classes,self.n_classes)) #Matrix of prediction vs true values

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
      with np.printoptions(linewidth = (10*self.n_classes+20),precision=4, suppress=True):
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
     









