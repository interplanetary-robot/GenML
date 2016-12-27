
# !((A & B) | (!A & !B))
#
# A B | A & B
# T T | T
# T F | F
# F T | F
# F F | F
#
# A B | !A & !B
# T T | F
# T F | F
# F T | F
# F F | T
#
# neural net layer 2[1] is going to be (A & B)
# neural net layer 2[2] is going to be (!A & !B)
#
# 2[1]: -7.5*bias + 5.0*A + 5.0*B
# 2[2]: 7.5*bias - 10.0*A - 10.0*B
#
# neural net layer 3[1] is going to be !(3[1] | 3[2])
# 3[1]: 5.0*bias + -10.0*A + -10.0*B
#
# A B | !(A | B)
# T T | F
# T F | F
# F T | F
# F F | T

#generate a multilayer perceptron
xornet = GenML.MLP.MultilayerPerceptron{Float64, (2,2,1)}()

#hand-written transition matrices.
xornet.layers[1].bias = [-7.5, 7.5]
xornet.layers[1].transition = [5.0 5.0; -10.0 -10.0]
xornet.layers[2].bias = [5.0]
xornet.layers[2].transition = [-10.0 -10.0]

@test xornet([true, true])[1] < 0.5
@test xornet([true, false])[1] > 0.5
@test xornet([false, true])[1] > 0.5
@test xornet([false, false])[1] < 0.5

#testing the allocations for evaluation.
dataset = rand(Bool, 2, 500)
results = Array{Float64}(1, 500)

GenML.batch_evaluate!(results, xornet, dataset, Val{500})
@time GenML.batch_evaluate!(results, xornet, dataset, Val{500})
