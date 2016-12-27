module MLP

import ..FCL.FullyConnectedLayer
import ..TF.sigmoid

import ..@import_interface
@import_interface

#a canned simple multilayer perceptron type
type MultilayerPerceptron{F, LayerDescriptor} <: MLAlgorithm{F}
  layers::Array{FullyConnectedLayer{F},1}
end

################################################################################
# constructors

#empty constructor - fills the MLP with zero transitions.  Useful for generating empty MLPs
(::Type{MultilayerPerceptron{F, LD}}){F, LD}() = MultilayerPerceptron{F, LD}(zeros)

#functions only constructor - fills MLP with zero transitions.
(::Type{MultilayerPerceptron{F, LD}}){F, LD}(fns::Array{Function, 1}) = MultilayerPerceptron{F, LD}(zeros, fns)

# rf - 'randomization function' must have the following property:
#   rf(F, n) -> 1-d array of random float F
#   rf(F, n, m) -> 2-d array of random float F.
# transfers - array of transferfunction.  must take F -> F.  An empty array means use the default layer transition
#   (should be sigmoid)
function (::Type{MultilayerPerceptron{F, LD}}){F, LD}(rf::Function, transfers::Array{Function, 1} = Function[])
  #double check that the transfers array has the same order as layer descriptor
  default_constructor = (length(transfers) == 0)
  if (!default_constructor)
    (length(transfers) != length(LD) - 1) && throw(ArgumentError("transfer function list must match length of the descriptors"))
    #test to make sure the transfer function successfully executes across arrays.
    for tf in transfers; testval = tf.(ones(F, 10)); end
  end

  #create an empty layers object.
  layers = FullyConnectedLayer{F}[]
  for idx = 1:(length(LD) - 1)
    if default_constructor
      push!(layers, FullyConnectedLayer{F, LD[idx], LD[idx + 1], sigmoid}(rf))
    else
      push!(layers, FullyConnectedLayer{F, LD[idx], LD[idx + 1], transfers[idx]}(rf))
    end
  end

  MultilayerPerceptron{F, LD}(layers)
end

#let's also define calling an MLAlgorithm, in cases where you don't need to agressively
#conserve memory allocations.
function (mlp::MultilayerPerceptron{F, LD}){F, LD}(input::Vector)
  #allocate an output vector.
  output = Vector{F}(LD[end])
  evaluate!(output, mlp, input)
  output
end

function (mlp::MultilayerPerceptron{F, LD}){F, LD}(input::Matrix)
  #allocate an output matrix.
  output = Matrix{F}(LD[end], size(input, 2))
  batch_evaluate!(output, mlp, input)
  output
end

include("./serialization.jl")
include("./evaluation.jl")
include("./backpropagation.jl")

#optionally, an MLAlgorithm can implement backpropagation.
#hasbackpropagation{T <: MLAlgorithm}(::Type{T}) = false  #default does not implement backpropagation
#backpropagate{F}(::MLAlgorithm{F}, training_input::Vector, training_solutions::Vector, costfn::Function, args...) = throw(MethodError(backpropagate, (MLAlgorithm{F}, Vector, Vector, Function)))
#backpropagate{F}(::MLAlgorithm{F}, training_input::Matrix, training_solutions::Matrix, costfn::Function, args...) = throw(MethodError(backpropagate, (MLAlgorithm{F}, Matrix, Matrix, Function)))

end
