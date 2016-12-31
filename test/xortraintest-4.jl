#xortraintest-5.jl

#trains an xornet using noisy and unreliable data.  Ten data inputs, top two xor'ed to get the
#correct values, 5% of the time.

function xortrain_4()

  srand(10)

  println("working on unreliable xor data set with backpropagation")

  input_matrix = rand(Bool, 10, 500)
  training_results = Array{Bool,2}(1,500)
  training_results[:] = [input_matrix[1, col] $ input_matrix[2, col] $ (rand() < 0.05) for col in 1:size(input_matrix,2)]

  xornet = GenML.MLP.MultilayerPerceptron{Float64,(10,2,1)}(randn)

  for rounds = 1:25
  for idx = 1:500
    input_column = input_matrix[:, idx]
    training_column = training_results[:, idx]

    GenML.Optimizers.backpropagationoptimize(xornet, input_column, training_column, GenML.CF.crossentropy)
  end
  end

  #verify that the optimization has resulted in a good data set.
  wrongcount = 0
  for x = 1:50
    input_vector = rand(Bool, 10)
    wrongcount += (xornet(input_vector)[1] > 0.5) != (input_vector[1] $ input_vector[2])
  end
  println("incorrect responses for noisy xor: $wrongcount")

  return wrongcount
end

xortrain_4() == 0 || xortrain_4() == 0 || throw(ErrorException("two executions failed."))
