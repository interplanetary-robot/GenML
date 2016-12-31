module MLP

import ..FCL.FullyConnectedLayer
import ..TF.sigmoid

import ..@import_interface
@import_interface

#a canned simple multilayer perceptron type
immutable MultilayerPerceptron{F, LayerDescriptor} <: MLAlgorithm{F}
  layers::Array{FullyConnectedLayer{F},1}
end

(mlp::MultilayerPerceptron{F, LD}){F,LD}(v::Array) = ml_call(mlp, v)

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

include("./storage.jl")
include("./serialization.jl")
include("./evaluation.jl")
include("./backpropagation.jl")

end
