from HPO.utils.files import load_obj
import os
import torch
import numpy as np
from HPO.utils.model_constructor import Model
from HPO.data.datasets import repsol_unlabeled
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from HPO.utils.worker_helper import collate_fn_padd_x
def load_model( model_code ):
	hyperparameter = load_obj( "/home/snaags/scripts/HPO/src/HPO/model_zoo/hps/{}".format(model_code) )
	model = Model((27, ) , 2 , hyperparameters = hyperparameter)
	model.load_state_dict(torch.load("/home/snaags/scripts/HPO/src/HPO/model_zoo/{}".format(model_code)))

	return model 

def generate_soft_labels( model ):
	batch_size = 1
	dataset = repsol_unlabeled()
	dataloader = DataLoader( dataset,
		shuffle = False,batch_size = 1,drop_last=False,pin_memory=True)
	output_list = []
	pred_list = []
	input_list = []
	model = model.cuda()
	with torch.no_grad(): #disable back prop to test the model
		model = model.eval()
		for i, (inputs) in enumerate( dataloader):

			inputs = inputs.cuda(non_blocking=True)
			outputs = model(inputs.float())

			preds = torch.argmax(outputs.view( batch_size ,2),1).long().cpu().numpy()

			output_list.append(outputs.cpu().numpy())
			input_list.append(inputs.cpu().numpy())
			pred_list.append(preds)

		y = np.asarray(output_list)
		input_list = np.asarray(input_list).reshape(27,-1)
		print(y.shape)
	dataset.add_labels(y)
	return dataset

if __name__ == "__main__":
	models = os.listdir("/home/snaags/scripts/HPO/src/HPO/model_zoo/")
	models = sorted(models)
	print(models[-2])
	model = load_model(models[-2])
	generate_soft_labels( model )
