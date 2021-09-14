
import torch
import torch.nn as nn
import numpy

def stdio_print_training_data( iteration : int , outputs : Tensor, labels : Tensor , epoch : int, epochs : int, acc :float , peak_acc : float , loss : Tensor, n_iter):
  def cal_acc(y,t):
      return np.count_nonzero(y==t)/len(y)

  def convert_label_max_only(y):
    y = y.cpu().detach().numpy()
    idx = np.argmax(y,axis = 1)
    out = np.zeros_like(y)
    for count, i in enumerate(idx):
        out[count, i] = 1
    return idx

  acc += cal_acc(convert_label_max_only(outputs), labels.cpu().detach().numpy())
  acc = acc/2
  if acc > peak_acc:
    peak_acc = acc
  # Save the canvas
  print("Epoch (",str(epoch),"/",str(epochs), ")""Iteration(s) (", str(iteration),"/",str(n_iter), ") Loss: "
    ,"%.2f" % loss, "Accuracy: ","%.2f" % acc ,"-- Peak Accuracy: %.2f" % peak_acc, end = '\r')
  return acc ,peak_acc


def train_model(model : Model , hyperparameters : dict, dataloader : DataLoader , epochs : int):
  n_iter =  int(train_dataset.get_n_samples()/batch_size)
  optimizer = torch.optim.Adam(model.parameters(),lr = hyperparameter["lr"])
  criterion = nn.CrossEntropyLoss()
  acc = 0
  peak_acc = 0
  for epoch in range(epochs)
    peak_acc = 0
    acc= 0
    for i, (samples, labels) in enumerate( dataloader ):
      samples = samples.cuda(non_blocking=True)
      labels = labels.cuda(non_blocking=True)
      # zero the parameter gradients
      optimizer.zero_grad()
      outputs = model(samples.float())
      # forward + backward + optimize
      outputs = outputs
      labels = labels.long()
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      if i%50 == 0:
        acc, peak_acc = stdio_print_training_data(i , outputs , labels, epoch,epochs , acc, peak_acc, loss.item(), n_iter)
