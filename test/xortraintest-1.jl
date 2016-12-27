#xortraintest-1.jl

#trains an xornet using noisy data.  Ten data inputs, top two xor'ed to get the
#correct values.

#encapsulate in a function so that the GC can take care of these variables.
function xortrain_1()

  println("working on noisy xor dataset")

  input_matrix = rand(Bool, 10, 500)
  expected_results = Array{Bool,2}(1,500)
  expected_results[:] = [input_matrix[1, col] $ input_matrix[2, col] for col in 1:size(input_matrix,2)]

  xornet = GenML.MLP.MultilayerPerceptron{Float64,(10,2,1)}(randn)

  GenML.Optimizers.gradientoptimize(xornet, input_matrix, expected_results, GenML.CF.crossentropy)

  #verify that the optimization has resulted in a good data set.
  wrongcount = 0
  for x = 1:50
    input_vector = rand(Bool, 10)
    wrongcount += (xornet(input_vector)[1] > 0.5) != (input_vector[1] $ input_vector[2])
  end
  println("incorrect responses for noisy xor: $wrongcount")

  return wrongcount
end

xortrain_1() == 0 || xortrain_1() == 0 || throw(ErrorException("two executions failed."))
