

def nats_encode(config): 
  paths = [[""],["",""],["","",""]]
  for i in config:
    paths[int(i[-1])-1][int(i[0])] =  config[i]+"~"+str(int(i[0]))
  for i in paths:
    paths[paths.index(i)] = "|"+"|".join(i)+"|"
  return "+".join(paths)
