class evaluator:
  def __init__(self, model , testloader, batch_size,n_classes):
    with torch.no_grad(): #disable back prop to test the model
      out_data = []
      self.batch_size = batch_size
      self.correct = 0
      self.incorrect= 0
      self.total = 0
      self.confusion_matrix = np.zeros(shape = (n_classes,n_classes))
      self.testloader = testloader
      self.model_prob = np.zeros(shape = (len(testloader), n_classes)) # [sample , classes]
      self.labels = np.zeros(shape = (len(testloader),1))
  def forward_pass(self):
      for i, (inputs, labels) in enumerate( testloader ):
          start_index = i * batch_size
          end_index = (i * batch_size) + batch_size
          inputs = inputs.cuda(non_blocking=True, device = cuda_device).float()
          self.labels[start_index:end_index , :] = labels.cpu().numpy()
          self.model_prob[start_index:end_index,:] = model(inputs).cpu().numpy()       

  def predictions(self, binary = False, THRESHOLD = None):
          if binary:
            self.prediction = np.where(self.model_prob > THRESHOLD, 1,0)
            assert self.prediction.shape == (len(testloader),1), "Shape of prediction is {} when it should be {}".format(self.prediction.shape, (len(testloader),1))
            self.correct = (self.predictions == self.labels)
          else:
            self.prediction = numpy.argmax(self.model_prob, axis = 1)
            assert self.prediction.shape == (len(testloader),1),  "Shape of prediction is {} when it should be {}".format(self.prediction.shape, (len(testloader),1))
            self.correct = (self.predictions == self.labels)
  
      
  def evaluate(self):
      for i, (inputs, labels) in enumerate( testloader):
          inputs = inputs.cuda(non_blocking=True, device = cuda_device).float()
          labels = labels.cuda(non_blocking=True, device = cuda_device).view( batch_size_test ).long().cpu().numpy()
          outputs = model(inputs).cuda(device = cuda_device)          
          if binary:
            print(outputs)
            for i in outputs:
              preds = (i > THRESHOLD)
              c = (preds == labels[0]).item()
              print("Reported Result {} -- Output: {} -- Prediction: {} -- Label: {}".format(c, outputs, preds ,labels))
          else:
            preds = torch.argmax(outputs.view(batch_size_test,n_classes),1).long().cpu().numpy()
            c = (preds == labels).sum()
  
          correct += c 
          t = len(labels)
          total += t
          for l,p in zip(labels, preds):
            if l == 1:
              recall_total += 1
              rt = 1 
              if l == p:
                rc = 1
                recall_correct += 1
              else:
                rc = 0
            else:
              rt = 0
          outputs = outputs.cpu().numpy()
