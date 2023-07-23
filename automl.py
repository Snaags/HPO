class AutoML:


  def train(self, dataset):
    """
    Training Interface: takes in a dataset object
    """
    pass

  def predict(self, dataset):
    """
    dataset : dataset.Dataset - only samples
    Called with: umodel.predict(dataset.get_split("test").remove_columns("label"))
    """
    pass
