#xortraintest-2.jl

#trains an xornet using noisy and unreliable data.  Ten data inputs, top two xor'ed to get the
#correct values, 5% of the time.

function xortrain_2()

  println("working on unreliable xor data set")

  input_matrix = rand(Bool, 10, 500)
  training_results = Array{Bool,2}(1,500)
  training_results[:] = [input_matrix[1, col] $ input_matrix[2, col] $ (rand() < 0.05) for col in 1:size(input_matrix,2)]

  xornet = GenML.MLP.MultilayerPerceptron{Float64,(10,2,1)}(randn)

  GenML.Optimizers.gradientoptimize(xornet, input_matrix, training_results, GenML.CF.crossentropy)

  #verify that the optimization has resulted in a good data set.
  wrongcount = 0
  for x = 1:50
    input_vector = rand(Bool, 10)
    wrongcount += (xornet(input_vector)[1] > 0.5) != (input_vector[1] $ input_vector[2])
  end
  println("incorrect responses for unreliable xor: $wrongcount")

  return wrongcount
end

xortrain_2() == 0 || xortrain_2() == 0 || throw(ErrorException("two executions failed."))
