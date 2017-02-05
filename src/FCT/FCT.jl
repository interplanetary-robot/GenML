#FCL.jl - an implementation of the "fully connected layer"

module FCT

import ..@import_interface
@import_interface

import ..TF.sigmoid

immutable FullyConnectedTransition{F, inputlength, outputlength, transferfunction} <: MLAlgorithm{F}
  bias::Vector{F}
  transition::Matrix{F}
end

(fcl::FullyConnectedTransition{F,i,o,tf}){F,i,o,tf}(v::Array) = ml_call(fcl, v)

function (::Type{FullyConnectedTransition{F,i,o,tf}}){F,i,o,tf}(randomizationfunction::Function = zeros)
  biasvector = randomizationfunction(F, o)
  transitionmatrix = randomizationfunction(F, o, i)
  FullyConnectedTransition{F,i,o,tf}(biasvector, transitionmatrix)
end

#implementation of the interfaces
inputs{F, i, o, tf}(::Type{FullyConnectedTransition{F, i, o, tf}}) = i
outputs{F, i, o, tf}(::Type{FullyConnectedTransition{F, i, o, tf}}) = o
parameters{F, i, o, tf}(::Type{FullyConnectedTransition{F, i, o, tf}})  = (i + 1) * o

include("./serialization.jl")
include("./evaluation.jl")
include("./backpropagation.jl")
include("./dropout.jl")

end #module
