module GenML

abstract MLAlgorithm{F} #general type for any machine learning algorithm.
abstract Layer{F} <: MLAlgorithm{F} #a generalized "layer" type

# interface definition for MLAlgorithm.

# basics.  input and output vector size count.
inputs{T <: MLAlgorithm}(::Type{T})     = throw(ArgumentError("inputs for $T not properly implemented"))
outputs{T <: MLAlgorithm}(::Type{T})    = throw(ArgumentError("outputs for $T not properly implemented"))
parameters{T <: MLAlgorithm}(::Type{T}) = throw(ArgumentError("parameters for $T not properly implemented"))

inputs{T <: MLAlgorithm}(::T)     = inputs(T)
outputs{T <: MLAlgorithm}(::T)    = outputs(T)
parameters{T <: MLAlgorithm}(::T) = parameters(T)

@generated function flatten!{F}(::Vector{F}, mla::MLAlgorithm{F}, offset::Integer = 0)
  T = mla
  :(throw(ArgumentError("flatten! for $T not propertly implemented")))
end
@generated function unflatten!{F}(mla::MLAlgorithm{F}, ::Vector{F}, offset::Integer = 0)
  T = mla
  :(throw(ArgumentError("unflatten! for $T not propertly implemented")))
end

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

#storage and backpropstorage for layer types is generally not allowed.
(::Type{Storage}){T <: Layer}(::Type{T})         = throw(ArgumentError("storage of layer types not allowed"))
(::Type{Storage}){T <: Layer}(::T)               = throw(ArgumentError("storage of layer types not allowed"))
(::Type{BackpropStorage}){T <: Layer}(::Type{T}) = throw(ArgumentError("storage of layer types not allowed"))
(::Type{BackpropStorage}){T <: Layer}(::T)       = throw(ArgumentError("storage of layer types not allowed"))

################################################################################
## evaluation

# MLAlgorithms should implement evaluate() methods.
@generated function evaluate!{F, bsize}(::AbstractArray{F}, mla::MLAlgorithm{F}, ::AbstractArray, ::Storage, ::Type{Val{bsize}} = Val{:auto})
  if mla <: Layer
    :(throw(ArgumentError("Layer types don't implement storage evaluation.")))
  else
    :(throw(ArgumentError("evaluate! not yet implemented")))
  end
end

@generated function evaluate!{F, bsize}(output::AbstractArray{F}, mla::MLAlgorithm{F}, input::AbstractArray, ::Type{Val{bsize}} = Val{:auto})
  #for the situation where the memory has not been preallocated, do the following.
  if input <: AbstractVector
    #catch mismatched input/output type parameters at the compile stage.
    (output <: AbstractVector) || throw(ArgumentError("vector/matrix status of input and output must match."))
    quote
      #generate the temporary storage
      sbuf = Storage(mla, :v)
      #go!
      evaluate!(output, mla, input, sbuf, Val{bsize})
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
      evaluate!(output, mla, input, sbuf, Val{bsize})
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

################################################################################
## backpropagation

#optionally, an MLAlgorithm can implement backpropagation.
hasbackpropagation{T <: MLAlgorithm}(::Type{T}) = false  #default does not implement backpropagation

#to give backpropagation algorithms the option of not returning the next deltas for the last layer.
typealias VoidableDeltas{F} Union{AbstractArray{F}, Void}

#compound ml algorithms with backpropagation should implement this function.
@generated function backpropagate!{F}(mla::MLAlgorithm{F},                   #algorithm to be backpropagated.
                                      inp::AbstractArray,                    #input values to be trained
                                      oup::AbstractArray{F},                 #expected output values for training
                                      ouδ::AbstractArray{F},                 #deltas for these output values
                                      sto::BackpropStorage,                  #backprop storage data.
                                                                             #input deltas for passing to the next layer.
                                      inδ::VoidableDeltas{F} = nothing)
  T = mla
  :(throw(ArgumentError("backpropagate! for $T not properly implemented.")))
end

#layers should implement backpropagation without storage function.
function backpropagate!{F}(mla::Layer{F},
                           inp::AbstractArray,
                           oup::AbstractArray{F},
                           ouδ::AbstractArray{F},
                           inδ::VoidableDeltas{F} = nothing)
  T = mla
  :(throw(ArgumentError("backpropagate! for $T not properly implemented.")))
end

################################################################################
## dropouts

abstract DropoutStorage{T <: MLAlgorithm}

#optionally an MLAlgorithm can implement dropouts
hasdropout{T <: MLAlgorithm}(::Type{T}) = false #which we set to as a default

generate_dropout_storage{T <: MLAlgorithm}(::T) = throw(ArgumentError("dropout not defined for type $T"))

doc"""
  GenML.dropout!(::MLAlgorithm, ::DropoutStorage)

  performs a dropout on the ML algorithm.  Values are stored in dropoutstorage for later reintegration using GenML.restore!
"""
dropout!{T <: MLAlgorithm}(mla::T, sto::DropoutStorage{T}) = throw(ArgumentError("dropout! not defined for type $T"))

doc"""
  GenML.restore!(::MLAlgorithm, ::DropoutStorage)

  restores lost values from a dropout! process.
"""
restore!{T <: MLAlgorithm}(mla::T, sto::DropoutStorage{T}) = throw(ArgumentError("restore! not defined for type $T"))

macro import_interface()
  quote
    #key functions in the interface
    import ..inputs; import ..outputs; import ..parameters; import ..flatten!; import ..unflatten!
    import ..evaluate!; import ..hasbackpropagation; import ..backpropagate!
    #key type definitions in the interface
    import ..MLAlgorithm; import ..Layer; import ..VoidableDeltas
    import ..Storage; import ..BackpropStorage; import ..DropoutStorage
    #dropout stuff
    import ..hasdropout; import ..generate_dropout_storage;
    import ..dropout!; import ..restore!
    #calling convenience function
    import ..ml_call
    #possibly useful modules.
    import ..TF; import ..CF
  end
end

#mathematical utilities
include("./math/math.jl")

#subtypes
include("./FCL/FCL.jl") #fully connected layer
include("./MLP/MLP.jl") #multilayer perceptrons

#include general optimizers
include("optimizers/optimizers.jl")

end # module
