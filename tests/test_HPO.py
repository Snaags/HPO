import pytest
from HPO.utils import weight_freezing
import torch.nn as nn
import torch
import numpy as np
from torch import Tensor
from HPO.utils.model_constructor import Model
from HPO.searchspaces.OOP_config import init_config as _config
from HPO.utils.worker_helper import train_model
from torch.utils.data import Dataset, DataLoader
import copy

class ExampleDataset(Dataset):
  def __init__(self, input_shape, output_size , num_samples):
    self.x = torch.rand( (num_samples , input_shape[1] ) )
    self.y = torch.randint(0,output_size, (num_samples, 1 )).squeeze() 
    self.input_shape = input_shape 
    self.window = input_shape[1]
    self.n_samples = num_samples - self.window
    self.n_classes = len(np.unique(self.y))

  def __getitem__(self, index):
    x = self.x[index:index+self.window]
    y = self.y[index+self.window-1]
    x = x.reshape(self.input_shape[0],self.window)
    return x, y
  
  def get_n_classes(self):
    return self.n_classes
  def get_n_samples(self):
    return self.n_samples

  def __len__(self):
    return self.n_samples 


@pytest.fixture(scope="session")
def input_shape():
    return (10,10)

@pytest.fixture(scope="session")
def output_size():
    return 2

@pytest.fixture(scope="session")
def input_data(input_shape : tuple):
    return torch.rand( ( input_shape ) )

@pytest.fixture(scope="session")
def labels( input_shape: tuple):

    return 
@pytest.fixture(scope="session")
def dataloader(input_shape : tuple, output_size):
    dataset = ExampleDataset(input_shape, output_size , num_samples = 10000)
    dataloader_var = DataLoader(dataset, batch_size = 50)
    return dataloader_var

@pytest.fixture(scope="session")
def hyperparameters(): 
    config = _config()
    hyperparameters = config.sample_configuration()
    yield hyperparameters.get_dictionary()

@pytest.fixture(scope="session")
def model( input_shape : tuple , output_size : int, hyperparameters):

    model = Model( input_shape[1:], output_size, hyperparameters )
    model = model.cuda()
    return copy.deepcopy(model)




def weight_clone( f ):
    def wrapper( *args , **kwargs):
        weights = f(*args , **kwargs)
        return [weight.detach().clone() for weight in weights]
    return wrapper


@weight_clone
def get_reduction_weights(model : Model ):
    for cell in model.reduction_cells:
        for mList in cell.compute_order:
            for operation in mList.compute:
                for param in operation.parameters():
                   yield param

@weight_clone
def get_normal_weights(model : Model ):
    for cell in model.normal_cells:
        for mList in cell.compute_order:
            for operation in mList.compute:
                for param in operation.parameters():
                   yield param

@weight_clone
def get_fc_weights(model : Model):
    for layer in model.fc_list:
        for param in layer.parameters():
            yield param

def mostly_true( bool_arr , threshold = 0.1):
    num_elements = 1
    for dim_size in bool_arr.size():
        num_elements *= dim_size
    if num_elements == 0:
        raise ValueError("Array size 0, either empty array or dims of size 0")
    true_vals = torch.sum(bool_arr)
    true_ratio = true_vals /num_elements
    print(true_ratio)
    return (true_ratio > threshold)


##Tests





def test_normal_weight_freezing( model : Model , dataloader : DataLoader , hyperparameters ):
    #Makes sure that the frozen normal weights are not being trained
    initial_weights = []
    hyperparameters["lr"] = 0.5 #Set high learning rate to encourage weights to change
    model_copy = copy.deepcopy(model)
    pre_training_weights = get_normal_weights(model)
    pre_training_weights_fc = get_fc_weights(model)
    print("Pre-training: ", pre_training_weights_fc[0])
    
    frozen_model = weight_freezing.freeze_normal_cells( model )

    train_model( frozen_model , hyperparameters , dataloader , epochs = 1 )
    train_model( model_copy ,hyperparameters , dataloader , epochs = 1 )
    
    print("saved weights after training : ", pre_training_weights_fc[0])
    post_training_frozen_model_weights = get_normal_weights(frozen_model)
    post_training_unfrozen_model_weights = get_normal_weights(model_copy)
    post_training_frozen_model_weights_fc = get_fc_weights(frozen_model)
    post_training_unfrozen_model_weights_fc = get_fc_weights(model_copy)
    print("Post Training weights: ", post_training_unfrozen_model_weights_fc[0])

    for pre_weights , post_frozen_weights ,post_unfrozen_weights in zip( pre_training_weights_fc , post_training_frozen_model_weights_fc , post_training_unfrozen_model_weights_fc):
       assert mostly_true(post_frozen_weights != post_unfrozen_weights)
       assert mostly_true(pre_weights != post_unfrozen_weights)
       assert mostly_true(pre_weights != post_frozen_weights )


    for pre_weights , post_frozen_weights ,post_unfrozen_weights in zip( pre_training_weights , post_training_frozen_model_weights , post_training_unfrozen_model_weights):
       assert mostly_true(pre_weights == post_frozen_weights )
       assert mostly_true(pre_weights != post_unfrozen_weights)
       assert mostly_true(post_frozen_weights != post_unfrozen_weights)




def test_reduction_weight_freezing( model : Model , dataloader : DataLoader , hyperparameters ):
    #Makes sure that the frozen reduction weights are not being trained
    initial_weights = []
    model_copy = copy.deepcopy(model)
    pre_training_weights = get_reduction_weights(model)


    frozen_model = weight_freezing.freeze_reduction_cells( model ) 
    train_model( frozen_model , hyperparameters , dataloader , epochs = 1 )
    train_model( model_copy ,hyperparameters , dataloader , epochs = 1 )
    post_training_frozen_model_weights = get_reduction_weights(frozen_model)
    post_training_unfrozen_model_weights = get_reduction_weights(model_copy)

    for pre_weights , post_frozen_weights ,post_unfrozen_weights in zip( pre_training_weights , post_training_frozen_model_weights , post_training_unfrozen_model_weights):
       assert mostly_true(pre_weights == post_frozen_weights )
       assert mostly_true(pre_weights != post_unfrozen_weights)
       assert mostly_true(post_frozen_weights != post_unfrozen_weights)

def test_reset_fc(  model : Model , dataloader : DataLoader , hyperparameters ):
    train_model( model ,hyperparameters , dataloader , epochs = 1 )
    train_weights = get_fc_weights( model )

    model.reset_fc(10)
    model.cuda()
    reset_weights = get_fc_weights( model )
    for train,reset in zip(train_weights,reset_weights):
        assert mostly_true(train != reset)


