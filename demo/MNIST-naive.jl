using MNIST
using GenML

srand(10)

#naive MNIST FCLNN demo, inspired by
# https://github.com/jostmey/DeepNeuralClassifier

#get the data.  It's already serialized as 784 x 60,000
data = traindata()
features = map(Float32, data[1] / 256)
feature_count = size(features,2)
labels = map(Int64, data[2])

#next, create the neural net.

scaled_rand(p...) = 0.1*randn(p...)

import GenML.TF: sigmoid, softplus, softmax

MNISTNET = GenML.MLP.MultilayerPerceptron{Float32, (784, 500, 500, 500, 10)}(scaled_rand, [sigmoid, softplus, softplus, softmax])
dropoutstorage = GenML.generate_dropout_storage(MNISTNET)

const N_minibatch = 100
minibatch_datastore =   Matrix{Float32}(784, N_minibatch)
minibatch_resultstore = Matrix{Bool}(10,  N_minibatch)
#allocate the backpropstorage for the minibatch.
minibatch_storage = GenML.BackpropStorage(MNISTNET, N_minibatch)
minibatch_hypotheses = Matrix{Float32}(GenML.outputs(MNISTNET), N_minibatch)
costfunction = GenML.CF.crossentropy
dcostfunction = GenML.Math.d(costfunction)

const N_updates = 100

for i = 1:N_updates

  #perform a round of dropouts.
  GenML.dropout!(MNISTNET, dropoutstorage)

  #assemble the minibatch.
  for idx = 1:N_minibatch
    batch_index = rand(1:feature_count)
    minibatch_datastore[:, idx] = features[:, batch_index]
    minibatch_resultstore[:, idx] = falses(10)
    minibatch_resultstore[labels[idx] + 1, idx] = true
  end

  #actually do the forward propagation.

  GenML.evaluate!(minibatch_hypotheses, MNISTNET, minibatch_datastore, minibatch_storage)

  res_deltas = (dcostfunction(minibatch_resultstore, minibatch_hypotheses))

  GenML.backpropagate!(MNISTNET, minibatch_datastore, minibatch_hypotheses, res_deltas, minibatch_storage)

  #restore the dropouts.
  GenML.restore!(MNISTNET, dropoutstorage)

  #test the network.
  GenML.evaluate!(minibatch_hypotheses, MNISTNET, minibatch_datastore, minibatch_storage)

  totals = 0
  for idx = 1:N_minibatch
    #check to see if the hypothesis matches the value.
    hyp_ind = indmax(minibatch_hypotheses[:, idx])
    #println("minibatch result:", minibatch_hypotheses[:, idx])
    totals += minibatch_resultstore[hyp_ind]
  end
  println("$totals % correct on batch $i")
end
