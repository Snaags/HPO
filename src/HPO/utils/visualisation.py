import matplotlib.pyplot as plt

def plot_scores(accuracy_scores):
      plt.plot(accuracy_scores)
      plt.title("Accuracy Scores of Configurations")
      plt.xlabel("Iteration")
      plt.ylabel("Accuracy")
      plt.grid()
      plt.savefig("livelogging.png",dpi = 800)
      plt.clf()



