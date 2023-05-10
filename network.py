import random
import math
import copy
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt


class Network:
  def __init__(self, shape,output_function='linear',activation_function='relu',scale=1,upper_b=10):
    self.shape = shape
    self.scale = scale

    self.upper_b = upper_b

    layers = []

    self.output_function = output_function
    self.activation_function = activation_function

    for i in range(len(shape)-1):
      layers.append(layer(shape[i], shape[i+1], output_function=self.activation_function, activation_function=self.activation_function,upper_b=self.upper_b))

    layers[-1].output_function = output_function
    layers[-1].scale = scale

    self.layers = layers


  def predict(self, network_inputs):

    layer_outputs = self.layers[0].predict(network_inputs)

    for l in range(1, len(self.layers)):
      layer_outputs = self.layers[l].predict(layer_outputs)

    return layer_outputs

  def birth(self, p):
    child = Network(self.shape,output_function=self.output_function,activation_function=self.activation_function)
    #child.layers = copy.deepcopy(self.layers)

    for l, layer in enumerate(self.layers):
      for b, bias in enumerate(layer.biases):
        child.layers[l].biases[b] = lerp(bias,random.uniform(-self.upper_b,self.upper_b), p)

      for ow, out_weights in enumerate(layer.weights):
        for c, connection in enumerate(out_weights):
          child.layers[l].weights[ow][c] = lerp(
              connection, random.uniform(-self.upper_b,self.upper_b) , p)

    return child
  
  def segmented_birth(self,p,rnge):
    child = Network(self.shape,output_function=self.output_function,activation_function=self.activation_function)
    #child.layers = copy.deepcopy(self.layers)

    for l in range(rnge[0],rnge[1]):
      layer = self.layers[l]
      for b, bias in enumerate(layer.biases):
        child.layers[l].biases[b] = lerp(bias, random.uniform(-self.upper_b,self.upper_b), p)

      for ow, out_weights in enumerate(layer.weights):
        for c, connection in enumerate(out_weights):
          child.layers[l].weights[ow][c] = lerp(
              connection, random.uniform(-self.upper_b,self.upper_b), p)

    return child
  
  def node_birth(self,p,lyr,node):
    weights = []
    bias = -1
    child = Network(self.shape, output_function=self.output_function,
                    activation_function=self.activation_function)
    
    child.layers[lyr].biases[node] = lerp(
        self.layers[lyr].biases[node], random.uniform(-self.upper_b,self.upper_b), p)
    
    for c, connection in enumerate(self.layers[lyr].weights[node]):
        child.layers[lyr].weights[node][c] = lerp(
            connection, random.uniform(-self.upper_b,self.upper_b), p)
        weights.append(child.layers[lyr].weights[node][c])
    
    return child, child.layers[lyr].biases[node],weights
        

  def save(self, filename):
    with open(filename, 'wb') as file:
      pickle.dump(self, file)

  def load(self, filename):
    with open(filename, 'rb') as file:
      return pickle.load(file)
  
  def validate(self,x,y):
    print('validating...')
    
    for i,x_0 in enumerate(x):
      print(f'{x_0} : {self.predict(x_0)[0]} : {y[i]}')

    print(f'loss : {self.loss(x,y)}')
  
  def copy(self): 
    cop = Network(self.shape,self.output_function,self.activation_function,self.scale,self.upper_b)
    cop.layers = copy.deepcopy(self.layers)
    return cop
  
  def get_bias(self,lyr,n):
    return self.layers[lyr].biases[n]
  
  def set_bias(self,lyr,n,nb):
    self.layers[lyr].biases[n] = nb
  
  def get_weights(self,lyr,n):
    return self.layers[lyr].weights[n]
  
  def set_weight(self,lyr,n,c,nw):
    self.layers[lyr].weights[n][c]=nw

  

  def fit(self, x, y, mutation_rate=0.1, population_size = 100, decay_rate=0.999, log=False, node_segmentation=False,generation_expansion_rate=1.1,dynamic_generations=False,back_prop=False,learning_rate=0.01,preciscion=0.1,persistence=3,num_epochs=100,max_size = 50,generalization_factor=0.00):
    parent = self.copy()


    history = []
    losses = [0]*num_epochs
    initial_rate = mutation_rate
    mutated = False
    
    plan = []
    for a in range(len(parent.layers)):
      for b in range(len(parent.layers[a].biases)):
        plan.append((a,b))


    for epoch in range(num_epochs):
      random.shuffle(plan)

      for tup in plan:
          
          lyr_idx,node_idx = tup[0],tup[1]

          test_x,test_y = self.trim(x,y,max_size)
          parent.perfect_bias(lyr_idx,node_idx,population_size,test_x,test_y,mutation_rate,generalization_factor,back_prop,learning_rate)
          for weight_idx in range(len(parent.layers[lyr_idx].weights[node_idx])):
            parent.perfect_weight(
                lyr_idx, node_idx, weight_idx, population_size, test_x, test_y,mutation_rate,generalization_factor,back_prop,learning_rate)
          #print(
              #f'node {node_idx + 1} in layer {lyr_idx+1} perfected to loss of {parent.loss(x,y)}')
      print(plan)  
      print(
          f'epoch : {epoch+1}/{num_epochs} loss : {parent.loss(x,y)}')

      # randomize to get out of ruts: 
      if epoch > 1 and parent.loss(x, y) == losses[epoch-1] and epoch != num_epochs - 1 and losses[epoch-1] == losses[epoch-2]:
        history.append((parent.layers, parent.loss(x, y)))
        mutation_rate *= 2
        mutated = True
      elif mutated:
        mutation_rate = initial_rate
        mutated = False

        #parent = parent.birth(mutation_rate * 0.01 * ((num_epochs - epoch)/num_epochs))
      
      losses[epoch] = parent.loss(x, y)

    # search history for better version if exists
    best_layers,best_loss = self.best_version(history)


    if best_loss < parent.loss(x,y):
      parent.layers = best_layers

    self.layers = parent.layers
    self.validate(x,y)

    plt.plot([x for x in range(num_epochs)],losses)
    plt.show()

  def best_version(self,history):
    b = float('inf')
    bl = -1

    for i in history:
      if i[1] < b:
        b = i[1] 
        bl = i[0]
    
    return bl,b
  
  def regress(self,x,y):
    x = np.array(x)
    y = np.array(y)
    return np.poly1d(np.polyfit(x,y,8)).deriv()

  def perfect_weight(self, lyr_idx, node_idx, weight_idx, population_size, x, y, p, g,back_prop,lr):

    if back_prop:
      # start with 10 values
      tester = self.copy()

      plt1 = []
      plt2 = []

      bound = -self.upper_b
      val = -bound
      vals = 100
      for _ in range(vals):

        tester.set_weight(lyr_idx, node_idx, weight_idx, val)
        plt1.append(val)
        plt2.append(tester.loss(x, y))

        val += (2*bound)/vals

      #plt.plot(plt1, plt2)
      
      w = self.get_weights(lyr_idx,node_idx)[weight_idx]
      dl_dw = self.regress(plt1, plt2)(w)
      nw = -(lr*dl_dw) + w
      self.set_weight(lyr_idx,node_idx,weight_idx,nw)

      # print('new weight: ', nw, 'poly loss :', np.poly1d(np.polyfit(np.array(plt1),
      #       np.array(plt2), 4))(nw), 'actual loss: ', self.loss(x, y), 'dl/dw :',dl_dw)

      # print(plt)
      return ()

    loss = self.loss(x, y)

    population = []
    s = set()

    for _ in range(population_size):
        seed = self.copy()
        weight = self.layers[lyr_idx].weights[node_idx][weight_idx]

        seed.layers[lyr_idx].weights[node_idx][weight_idx] = lerp(
            weight, random.uniform(-self.upper_b, self.upper_b), p)
        
        population.append((seed,seed.loss(x,y)))

        if not g:
          if seed.loss(x, y) < loss:
            loss = seed.loss(x, y)
            self.layers = seed.layers
    if g:
      # remove synonymous seeds
      population = [t for t in population if t[1] not in s and not s.add(t[1])]
      
      idx = int(len(population)*g)

      self.layers = sorted(population, key=lambda x: x[1])[idx][0].layers

  def perfect_bias(self,lyr_idx,node_idx,population_size,x,y,p,g,back_prop,lr):
    if back_prop:
      # start with 10 values
      tester = self.copy()

      plt1 = []
      plt2 = []

      bound = -self.upper_b
      val = -bound
      vals = 100

      for _ in range(vals):

        tester.set_bias(lyr_idx, node_idx, val)

        plt1.append(val)
        plt2.append(tester.loss(x, y))

        val += (2*bound)/vals

      
      #plt.plot(plt1, plt2, 'r--')
      
      # print(plt)
      b = self.get_bias(lyr_idx,node_idx)
      dl_db = self.regress(plt1, plt2)(b)
      nb = b -(lr*dl_db)
      self.set_bias(lyr_idx, node_idx, nb)
      
      return ()

    loss = self.loss(x,y)

    population = []
    s = set()

    for _ in range(population_size):
        seed = self.copy()
        bias = self.layers[lyr_idx].biases[node_idx]
        seed.layers[lyr_idx].biases[node_idx] = lerp(bias,random.uniform(-self.upper_b,self.upper_b),p)

        population.append((seed,seed.loss(x,y)))

        if not g:
          if seed.loss(x,y) < loss:
            loss = seed.loss(x,y)
            self.layers = seed.layers
    
    if g:
      # remove synonymous seeds
      population = [t for t in population if t[1] not in s and not s.add(t[1])]
      idx = int(len(population)*g)
      self.layers = sorted(population, key=lambda x: x[1])[idx][0].layers

  def trim(self,x,y,mx_size):
    if len(x) <= mx_size:
      return x,y
    
    sx,sy=[],[]
    for _ in range(len(x)):
      r = random.randint(len(x)-1)
      sx.append(x[r])
      sy.append(y[r])

    return sx[0:mx_size],sy[0:mx_size]

    
  def loss(self, x, y):
    loss = 0
    for j, i in enumerate(x):
        y1 = self.predict(i)[0]

        try :
          loss += ((y1-y[j])**2)
        except:
          return 9999999999

    return loss


class layer:
  def __init__(self, inp, out, output_function, activation_function, upper_b,scale=1):
   self.scale = scale

   self.inp = inp
   self.out = out

   self.output_function = output_function
   self.activation_function = activation_function

   self.upper_b = upper_b

   self.biases, self.weights = self.generate_layer_contents()

   

  def generate_layer_contents(self):

    bs = []
    for i in range(self.out):
      bs.append(random.uniform(-self.upper_b,self.upper_b))

    ws = []
    for i in range(self.out):
      out_weights = []
      for j in range(self.inp):
        out_weights.append(random.uniform(-self.upper_b, self.upper_b))
      ws.append(out_weights)

    return bs, ws

  def predict(self, layer_inputs):
    layer_outputs = []
    for j in range(self.out):
      act = 0
      for i in range(self.inp):
        i_n = layer_inputs[i]

        if self.activation_function == 'relu':
          i_n = relu(i_n)
        if self.activation_function == 'sigmoid':
          i_n = sigmoid(i_n)

        act += (i_n*self.weights[j][i]) 
        #act -= (self.weights[j][i]**2)

      if self.output_function == 'relu':
        layer_outputs.append(self.scale*relu(act-self.biases[j]))
      elif self.output_function == 'sigmoid':
        layer_outputs.append(self.scale*sigmoid(act-self.biases[j]))
      elif self.output_function == 'linear':
        layer_outputs.append(self.scale*(act-self.biases[j]))
      else:
        return None

    return layer_outputs
  
def sigmoid(x):
  return 1 - (1 / (1 + np.exp(x)))


def relu(x):
  if x < 0:
    return 0

  return x


def lerp(a, b, t):
  return ((b-a)*t) + a


def test():
  net1 = Network([3, 5, 5, 1],output_function='relu',scale=10)
  net2 = Network([1, 5, 5, 1])
  net3 = net1.birth(0.01)
  net4 = net1.birth(1)
  net5 = net1.birth(0.01)

  print('random net 1 with input 1', net1.predict([1, 1, 4]))
  print('random net 2 wiht input 1', net2.predict([1]))
  print('net 3 birthed from 1 with 0.01', net3.predict([1, 1, 4]))
  print('net 5 birthed from 1 with 0.01', net5.predict([1, 1, 4]))
  print('net 4 birthed from 1 with 0.1', net4.predict([1, 1, 4]))
  print('random net 1 (unmutated by birth?)', net1.predict([1, 1, 4]))
  print('random net 1 one to one? ', net1.predict([-1, 0, 1]))

  #net1.save('./network_save.txt')
  net6 =  Network([3, 5, 1],output_function='sigmoid')
  print('pre load: ', net6.predict([1, 1, 4]))
  #net6 = net6.load('./network_save.txt')
  print('post load', net6.birth(0).predict([1, 1, 4]))


test()
