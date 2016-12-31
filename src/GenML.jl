module GenML

abstract MLAlgorithm{F} #general type for any machine learning algorithm.
abstract Layer{F} <: MLAlgorithm{F} #a generalized "layer" type

# interface definition for MLAlgorithm.

# basics.  input and output vector size count.
inputs{T <: MLAlgorithm}(::Type{T})     = throw(MethodError(inputs, (Type{T},)))
outputs{T <: MLAlgorithm}(::Type{T})    = throw(MethodError(outputs, (Type{T},)))
parameters{T <: MLAlgorithm}(::Type{T}) = throw(MethodError(parameters, (Type{T},)))

inputs{T <: MLAlgorithm}(::T)     = inputs(T)
outputs{T <: MLAlgorithm}(::T)    = outputs(T)
parameters{T <: MLAlgorithm}(::T) = parameters(T)


flatten!{F}(::Vector{F}, ::MLAlgorithm{F}, offset::Integer = 0) = throw(MethodError(flatten!, (Vector{F}, MLAlgorithm{F}, Integer)))
unflatten!{F}(::MLAlgorithm{F}, ::Vector{F}, offset::Integer = 0) = throw(MethodError(unflatten!, (MLAlgorithm{F}, Vector{F}, Integer)))

################################################################################
## nonparametric data storage.

doc"""
  `GenML.Storage(::Type{MLAlgorithm}, N = :v)`
  `GenML.Storage(::MLAlgorithm, N = :v)`

  takes an ML algorithm and allocates all non-parametric memory required to
  evaluate the algorithm.
"""
abstract Storage{T <: MLAlgorithm, N}

doc"""
  `GenML.BackpropStorage(::Type{MLAlgorithm})`
  `GenML.BackpropStorage(::MLAlgorithm)`

  takes an ML algorithm and allocates all non-parametric memory required to
  evaluate AND backpropagate the algorithm.  Implementations of backprop storage
  should be written in such a way that it be passed as a parameter to the
  forward evaluation which will persistently populate the layer data for the
  backwards propagation.
"""
abstract BackpropStorage{T, N} <: Storage{T, N}

################################################################################
## evaluation

# MLAlgorithms should implement evaluate() methods.
@generated function evaluate!{F}(::AbstractArray{F}, mla::MLAlgorithm{F}, ::AbstractArray, ::Storage)
  if mla <: Layer
    :(throw(ArgumentError("Layer types don't implement storage evaluation.")))
  else
    :(throw(MethodError(evaluate!, (AbstractArray{F}, MLAlgorithm{F}, Array, Storage))))
  end
end

@generated function evaluate!{F}(output::AbstractArray{F}, mla::MLAlgorithm{F}, input::AbstractArray)
  #for the situation where the memory has not been preallocated, do the following.
  if input <: AbstractVector
    #catch mismatched input/output type parameters at the compile stage.
    (output <: AbstractVector) || throw(ArgumentError("vector/matrix status of input and output must match."))
    quote
      #generate the temporary storage
      sbuf = Storage(mla, :v)
      #go!
      evaluate!(output, mla, input, sbuf)
    end
  elseif input <: AbstractMatrix
    #catch mismatched input/output type parameters at the compile stage.
    (output <: AbstractMatrix) || throw(ArgumentError("vector/matrix status of input and output must match."))
    quote
      #do a second check to make sure the size of the batches are the same.
      size(output, 2) == size(input, 2) || throw(ArgumentError("input and output matrices must have matching batch sizes"))
      #generate the temporary storage with the correct batch size.
      sbuf = Storage(mla, size(input, 2))
      #go!
      evaluate!(output, mla, input, sbuf)
    end
  end
end

@generated function ml_call{F}(mla::MLAlgorithm{F}, input::Array)
  if input <: Vector
    quote
      output = Array{F,1}(outputs(mla))
      evaluate!(output, mla, input)
      output
    end
  elseif input <: Matrix
    quote
      output = Array{F,2}(outputs(mla), size(input, 2))
      evaluate!(output, mla, input)
      output
    end
  end
end

#optionally, an MLAlgorithm can implement backpropagation.
hasbackpropagation{T <: MLAlgorithm}(::Type{T}) = false  #default does not implement backpropagation
backpropagate!{F}(::MLAlgorithm{F}, input_values::Vector{F}, output_deltas::Vector{F}) = throw(MethodError(backpropagate, (MLAlgorithm{F}, Vector, Vector)))


macro import_interface()
  quote
    #key functions in the interface
    import ..inputs; import ..outputs; import ..parameters; import ..flatten!; import ..unflatten!
    import ..evaluate!; import ..hasbackpropagation; import ..backpropagate!
    #key type definitions in the interface
    import ..MLAlgorithm; import ..Layer; import ..Storage; import ..BackpropStorage
    #calling convenience function
    import ..ml_call
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
