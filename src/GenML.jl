module GenML

abstract MLAlgorithm{F} #general type for any machine learning algorithm.
abstract Layer{F} <: MLAlgorithm{F} #a generalized "layer" type

# interface definition for MLAlgorithm.

# basics.  input and output vector size count.
inputs{T <: MLAlgorithm}(::Type{T}) = throw(MethodError(inputs, (Type{T},)))
outputs{T <: MLAlgorithm}(::Type{T}) = throw(MethodError(outputs, (Type{T},)))
parameters{T <: MLAlgorithm}(::Type{T}) = throw(MethodError(parameters, (Type{T},)))

flatten!{F}(::Vector{F}, ::MLAlgorithm{F}, offset::Integer = 0) = throw(MethodError(flatten!, (Vector{F}, MLAlgorithm{F}, Integer)))
unflatten!{F}(::MLAlgorithm{F}, ::Vector{F}, offset::Integer = 0) = throw(MethodError(unflatten!, (MLAlgorithm{F}, Vector{F}, Integer)))

# MLAlgorithms should implement evaluate() methods.
evaluate!{F}(::Vector{F}, ::MLAlgorithm{F}, ::Vector) = throw(MethodError(evaluate, (Vector{F}, MLAlgorithm{F}, Vector)))
batch_evaluate!{F,N}(::Matrix{F}, ::MLAlgorithm{F}, ::Matrix, ::Type{Val{N}} = Val{0}) = throw(MethodError(batch_evaluate!, (Matrix{F}, MLAlgorithm{F}, Matrix, Type{Val{N}})))

#optionally, an MLAlgorithm can implement backpropagation.
hasbackpropagation{T <: MLAlgorithm}(::Type{T}) = false  #default does not implement backpropagation
backpropagate!{F}(::MLAlgorithm{F}, input_values::Vector{F}, output_deltas::Vector{F}) = throw(MethodError(backpropagate, (MLAlgorithm{F}, Vector, Vector)))

#backpropagate{F}(::MLAlgorithm{F}, training_input::Vector, training_solutions::Vector, costfn::Function, args...) = throw(MethodError(backpropagate, (MLAlgorithm{F}, Vector, Vector, Function)))
#backpropagate{F}(::MLAlgorithm{F}, training_input::Matrix, training_solutions::Matrix, costfn::Function, args...) = throw(MethodError(backpropagate, (MLAlgorithm{F}, Matrix, Matrix, Function)))

macro import_interface()
  quote
    #key functions in the interface
    import ..inputs; import ..outputs; import ..parameters; import ..flatten!; import ..unflatten!
    import ..evaluate!; import ..batch_evaluate!; import ..hasbackpropagation; import ..backpropagate!
    #key type definitions in the interface
    import ..MLAlgorithm; import ..Layer
    #possibly useful modules.
    import ..TF; import ..CF
    #differentiation interface
    import ..d; import ..dxasy
  end
end

#mathematical utilities
#tools for differentiation
include("./differentiation.jl")
#transfer functions.
include("./transferfunctions.jl")
#include general cost functions
include("./costfunctions.jl")

#subtypes
include("./FCL/FCL.jl") #fully connected layer
include("./MLP/MLP.jl") #multilayer perceptrons

#include general optimizers
include("optimizers/optimizers.jl")

end # module
