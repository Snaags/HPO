# -*- coding: utf-8 -*-
"""
Pytorch triplet network functions

Created on Thu Jun 16 12:20:41 2022

@author: jackw

Adapted from:
    https://github.com/omoindrot/tensorflow-triplet-loss/blob/master/model/triplet_loss.py
    
"""

import torch as T
import time


# ---------------------------- Helper Functions -------------------------------

# calculate euclidean distance matrix for online mining 
def distance_matrix(embeddings, squared = True):
    
    #find the dot product matrix for embeddings
    dot_product = T.matmul(embeddings,T.transpose(embeddings, 0, 1))

    #get digonals from dp matrix (squared sum of embedding row)
    sq_sum = T.diagonal(dot_product, 0)

    
    '''
    calculate the distance matrix using: 
        ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
    
    shape is shape (batch_size, batch_size)
    '''
    
    d =  T.unsqueeze(sq_sum, 0) - 2.0 * dot_product + T.unsqueeze(sq_sum, 1)
    d = T.clamp(d, min = 0.0)  #prevent negatives due to floating point errors
    
    
    #sqrt distance matrix if squared distance not required
    if not squared:
        mask = T.eq(d, T.tensor(0.0)).float()  #find zero value s
        d = d + (mask * 1e-16)  #replace with small epsilon value

        d = T.sqrt(d)  #sqrt distance matrix
        
        d = d * (1.0 - mask)  #set epsilon values back to 0.0

    return d


#get masks for triplet mining
def _get_triplet_mask(labels, device = 'cpu'):
    """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
    A triplet (i, j, k) is valid if:
        - i, j, k are distinct
        - labels[i] == labels[j] and labels[i] != labels[k]
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    """
    
    # Check that i, j and k are distinct
    I_eq = T.eye(labels.shape[0], dtype = T.bool).to(device)  #get matrix where indices are equal (BxB boolean identity matrix)
    I_neq = T.logical_not(I_eq)  #invert matrix to find where indices are not equal 
    
    z_neq_r = T.unsqueeze(I_neq, 2)  #find where z index does not equal row index
    z_neq_c =T.unsqueeze(I_neq, 1)  #find where z index does not equal column index
    r_neq_c = T.unsqueeze(I_neq, 0)  #find where row index does not equal column index
                                
    distinct_mask = T.logical_and(T.logical_and(z_neq_r, z_neq_c), r_neq_c)  #get mask showing true for triplets with 2 unique samples


    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_eq = T.eq(T.unsqueeze(labels, 0), T.unsqueeze(labels, 1))  #get 2d matric of showing location of equal labels
    z_eq_r = T.unsqueeze(label_eq, 2)  #convert to column matrices
    z_eq_c = T.unsqueeze(label_eq, 1)  #convert to row matrices

    label_mask = T.logical_and(z_eq_r, T.logical_not(z_eq_c))  #create 3d mask of triplets with valid label combinations

    # Combine the two masks
    mask = T.logical_and(distinct_mask, label_mask)

    return mask


def _get_anchor_positive_triplet_mask(labels, device = 'cpu'):
    """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check that i and j are distinct
    I_eq = T.eye(labels.shape[0], dtype = T.bool).to(device)  #get matrix where indices are equal (BxB boolean identity matrix)
    I_neq = T.logical_not(I_eq)  #invert matrix to find where indices are not equal 

    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_eq = T.eq(T.unsqueeze(labels, 0), T.unsqueeze(labels, 1))

    # Combine the two masks
    mask = T.logical_and(I_neq, labels_eq)

    return mask


def _get_anchor_negative_triplet_mask(labels):
    """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
    Args:
        labels: tf.int32 `Tensor` with shape [batch_size]
    Returns:
        mask: tf.bool `Tensor` with shape [batch_size, batch_size]
    """
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_eq = T.eq(T.unsqueeze(labels, 0), T.unsqueeze(labels, 1))

    mask = T.logical_not(labels_eq)

    return mask


#-------------------------- Online Triplet Mining -----------------------------

#Online Triplet mining loss using batch all strategy
class Batch_All_Triplet_Loss(T.nn.Module):
  def __init__(self, m= 1.0, metric_matrix = distance_matrix, squared = True, device = 'cpu'):
    super(Batch_All_Triplet_Loss, self).__init__()  # pre 3.3 syntax
    
    self.m = m  # margin or radius
    self.metric_matrix = metric_matrix
    self.squared = squared
    self.fraction_positive_triplets = 0
    self.device = device
    #self.class_hist = []

  def get_fraction_pos(self):
     return self.fraction_positive_triplets
  
    
  def get_class_hist(self):
      return self.class_hist
      
  def forward(self, embeddings, target):
    """Build the triplet loss over a batch of embeddings.

    We generate all the valid triplets and average the loss over the positive ones.

    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.

    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    class_counts = [0 for x in range(5)]
    
    # Get the pairwise distance matrix
    d_matrix = self.metric_matrix(embeddings, squared = self.squared )
    
    #compute BxBxB distance matrices for anchor to positve and anchor to negative samples
    d_ap_matrix = T.unsqueeze(d_matrix, 2) 
    d_an_matrix = T.unsqueeze(d_matrix, 1)
    #return d_matrix
    # Compute BxBxB Triplet loss matrix
    # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
    # Uses broadcasting where the 1st argument has shape (B, B, 1)
    # and the 2nd (B, 1, B)
    triplet_loss = d_ap_matrix - d_an_matrix + self.m
    
    # zero loss for invalid triplets
    # (where label(a) != label(p) or label(n) == label(a) or a == p)
    
    mask = _get_triplet_mask(target, device= self.device).float()  #get mask for valid triplets
    
    triplet_loss = mask * triplet_loss  #apply mask to triplet loss matrix
    
    triplet_loss = T.clamp(triplet_loss, min=0.0)  #remove negative losses (easy triplets)
    
    '''
    indicies = T.nonzero(triplet_loss)  
    unique = np.array([T.unique(i).detach().numpy() for i in indicies]).flatten()
    indicies, counts = np.unique(unique, return_counts = True)
    
    for c, i in enumerate(indicies):
        class_counts[target[i]] += counts[c]
    
    self.class_hist.append(class_counts)
    '''
    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = T.greater(triplet_loss,1e-16).float()
    num_positive_triplets = T.sum(valid_triplets)
    num_valid_triplets = T.sum(mask)
    self.fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    #print(num_positive_triplets)
    #print(num_valid_triplets)

    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = T.sum(triplet_loss) / (num_positive_triplets + 1e-16)


    return triplet_loss

    
  


#semi hard triplet mining
class  Batch_All_Semi_Hard_Triplet_Loss(T.nn.Module):
    def __init__(self, m= 1.0, metric_matrix = distance_matrix, squared = True, device = 'cpu'):
      super(Batch_All_Semi_Hard_Triplet_Loss, self).__init__()  # pre 3.3 syntax
      self.m = m  # margin or radius
      self.metric_matrix = metric_matrix
      self.squared = squared
      self.fraction_positive_triplets = 0
      self.device = 'cpu'
    
    def forward(self, embeddings, target):
        """Build the triplet loss over a batch of embeddings for semi hard triplets.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        
        d_matrix = self.metric_matrix(embeddings, squared= self.squared)
        
        mask_triplet = _get_triplet_mask(target).float()  #get mask for valid triplets
        
        
        
        d_ap = T.unsqueeze(d_matrix, 2)  # col matrices of anchor positive distances

        d_an = T.unsqueeze(d_matrix, 1)  # row matrices of anchor positive distances

        mask_sh = T.gt(d_an,d_ap).float()  #mask filtering out non semihard triplets
        
        
        #calculate triplet loss
        triplet_loss = d_ap - d_an + self.m  #find triplet loss for all triplets
        triplet_loss = triplet_loss * mask_triplet * mask_sh  #apply masks to filter invalid triplets
        triplet_loss = T.mean(T.clamp(triplet_loss, min=0.0))  #remove negative losses (easy triplets) and find mean loss
        
        
        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = T.greater(triplet_loss,1e-16).float()
        num_positive_triplets = T.sum(valid_triplets)
        
        num_valid_triplets = T.sum(mask_triplet)
        self.fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)



        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = T.sum(triplet_loss) / (num_positive_triplets + 1e-16)
        
        
        return triplet_loss  #return triplet loss


    def get_fraction_pos(self):
        return self.fraction_positive_triplets


#semi hard triplet mining
class  Batch_Semi_Hard_Triplet_Loss(T.nn.Module):
    def __init__(self, m= 1.0, metric_matrix = distance_matrix, squared = True, device = 'cpu'):
      super(Batch_Semi_Hard_Triplet_Loss, self).__init__()  # pre 3.3 syntax
      self.m = m  # margin or radius
      self.metric_matrix = metric_matrix
      self.squared = squared
      self.device = device
    
    def forward(self, embeddings, target):
        """Build the triplet loss over a batch of embeddings for semi hard triplets.
        For each anchor, we get the positive and negative combination with the greatest loss where D_an > D_ap
        .
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        
        d_matrix = self.metric_matrix(embeddings, squared= self.squared)
        
        mask_triplet = _get_triplet_mask(target, device = self.device).float()  #get mask for valid triplets
        
        
        
        d_ap = T.unsqueeze(d_matrix, 2)  # col matrices of anchor positive distances

        d_an = T.unsqueeze(d_matrix, 1)  # row matrices of anchor positive distances

        mask_sh = T.gt(d_an,d_ap).float()  #mask filtering out non semihard triplets
        
        
        #calculate triplet loss
        triplet_loss = d_ap - d_an + self.m  #find triplet loss for all triplets
        triplet_loss = triplet_loss * mask_triplet * mask_sh  #apply masks to filter invalid triplets
        triplet_loss = T.amax(triplet_loss, dim=(1, 2))  #get hardest semihard triplet for each anchor
        triplet_loss = T.mean(T.clamp(triplet_loss, min=0.0))  #remove negative losses (easy triplets) and find mean loss

        return triplet_loss  #return triplet loss




#Online Triplet mining loss using batch hard strategy
class Batch_Hard_Triplet_Loss(T.nn.Module):
  def __init__(self, m= 1.0, metric_matrix = distance_matrix, squared = True, device = 'cpu'):
    super(Batch_Hard_Triplet_Loss, self).__init__()  # pre 3.3 syntax
    
    self.m = m  # margin or radius
    self.metric_matrix = metric_matrix
    self.squared = squared
    self.device = device

  def forward(self, embeddings, target):
    """Build the triplet loss over a batch of embeddings.
    For each anchor, we get the hardest positive and hardest negative to form a triplet.
    Args:
        labels: labels of the batch, of size (batch_size,)
        embeddings: tensor of shape (batch_size, embed_dim)
        margin: margin for triplet loss
        squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                 If false, output is the pairwise euclidean distance matrix.
    Returns:
        triplet_loss: scalar tensor containing the triplet loss
    """
    
    # Get the pairwise distance matrix
    d_matrix = self.metric_matrix(embeddings, squared= self.squared)


    # Get hardest positive sample for each anchor
    mask_ap = _get_anchor_positive_triplet_mask(target, device = self.device).float()  #get mask for positive samples for each anchor
    
    #print(d_matrix)
    #print(mask_ap)
    d_ap = mask_ap * d_matrix  # zero distance for samples with labels which do not match anchor
    d_hard_p = T.max(d_ap, dim = 1, keepdim = True).values  # get maximum distance between anchor and positive samples, for each anchor
    
                        
    # Get hardest negative sample for each anchor
    mask_an = _get_anchor_negative_triplet_mask(target).float()
    d_max_an = T.max(d_matrix, dim=1, keepdim=True).values  #get max distance betweene each anchor and negative sample
    d_an = d_matrix + (d_max_an * (1.0 - mask_an))  #add max distance to invalid pairs to ensure they are not selected for triplet (cannot 0 as they would be selected)

    d_hard_n = T.min(d_an, dim = 1, keepdim = True).values

    triplet_loss = T.clamp(d_hard_p - d_hard_n + self.m, min = 0.0)  # calculate triplet loss using hardest samples for each anchor

    triplet_loss = T.mean(triplet_loss)  # get final mean triplet loss

    return triplet_loss



#--------------------------- Custom Training Loop -----------------------------
'''
def Triplet_Train(model, optimiser, loss_fn, train_dl, epochs, val_dl = None, ep_log_interval = 1, device = 'cpu'):
    print('Training Model')

    #coolect training and validation metrics for each epoch
    history = {}
    history['loss'] = []
    history['val_loss'] = []    
    
    start_time = time.time()
    
    #start training loop
    for epoch in range(epochs):
        
        train_loss = 0  #initalise train loss
        
        model.train()  #set model to train
        
        #--------------- train and evaluate on training dataset ---------------
        
        for batch_num, batch in enumerate(train_dl):
           
            x_a, x_p, x_n = batch  #get validation batch
            

            emb_a, emb_p, emb_n = model(x_a, x_p, x_n)
            #emb_p = model(x_p)
            #emb_n = model(x_n)  #predict euclidean distances for validation batch
            
            optimiser.zero_grad()  #reset gradient values
            
            loss = loss_fn(emb_a, emb_p, emb_n)  #compute training loss
            
            #apply backpropogation
            loss.backward()         # compute gradients 
            optimiser.step()            # update weights
            
            train_loss += loss.item() * x_a.size(0)  # multiply sample loss by batch size for batch loss
            
        train_loss = train_loss / len(train_dl.dataset)  #find per sample loss
        
        
        #-------------------- evaluate on training dataset --------------------
        model.eval()  #set model to evaluat
        val_loss = 0  #init val loss
        
        if val_dl is not None:
            for batch_num, batch in enumerate(val_dl):
            
                x_a, x_p, x_n = batch  #get validation batch
                
                emb_a, emb_p, emb_n = model(x_a, x_p, x_n)
                #emb_a = model(x_a) 
                #emb_p = model(x_p) 
                #emb_n = model(x_n)  #predict euclidean distances for validation batch
                
                loss = loss_fn(emb_a, emb_p, emb_n)  #compute validation loss
                
                val_loss += loss.item() * x_a.size(0)  # multiply sample loss by batch size for batch loss
                
            val_loss = val_loss / len(val_dl.dataset)
            
        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if epoch % ep_log_interval == 0 and ep_log_interval != 0:
            train_time = time.time() - start_time
            print(f"epoch = {epoch}  |  train_loss = {train_loss:.6f}  |  val_loss = {val_loss:.6f} |  training_for: {train_time:.2f}" )
    
    
    
    #end model training
    end_time = time.time()
    total_time = end_time - start_time  #find time to trian
    #time_per_epoch = total_time / epochs  #find time per epoch
    
    print(f'\nTraining Complete in: {total_time:.2f} seconds')
    
    return history
'''