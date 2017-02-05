#TFL.jl - an implementation of a "dumb transfer transition"

module TFT

import ..@import_interface
@import_interface

doc"""
  GenML.TFT.TransferTransition{F, layerlength, transferfunction}

  is a machine learning algorithm that is a "dumb transfer transition."

  initalize as follows:

  GenML.TFT.TransferTransition{F, length, fn}()
"""
immutable TransferTransition{F, layerlength, transferfunction} <: MLAlgorithm{F}; end

(tft::TransferTransition{F,l,tf}){F,i,o,tf}(v::AbstractArray) = ml_call(tft, v)

#implementation of the interfaces
inputs{F,l,tf}(::Type{TransferTransition{F,l,tf}}) = l
outputs{F,l,tf}(::Type{TransferTransition{F,l,tf}}) = l
parameters{F,l,tf}(::Type{TransferTransition{F,l,tf}}) = 0

#flatten! and unflatten! don't do anything for transfertransitions since they have no parameters.
flatten!{F,l,tf}(v::Vector{F}, tft::TransferTransition{F,l,tf}, offset::Integer = 0) = nothing
unflatten!{F,l,tf}(tft::TransferTransition{F,l,tf}, v::Vector{F}, offset::Integer = 0) = nothing

#evaluation, for vectors.
@generated function evaluate!{F,l,tf,bsize}(output::AbstractVector{F},
  tft::TransferTransition{F,l,tf}, input::AbstractVector, ::Type{Val{bsize}} = Val{:auto})

  (bsize != :v) && (bsize != :auto) && return :(throw(ArgumentError("vector evaluation can't do a numerical bsize.")))

  #some functions, like softmax, can't be unrolled.  Other functions can be.
  if nounroll(tf)
    :(output[:] = tf(output))
  else
    quote
      for idx = 1:o
        output[idx] = tf(output[idx])
      end
    end
  end
end

#evaluation, for matrices.
@generated function evaluate!{F,l,tf,bsize}(output::AbstractMatrix{F},
  tft::TransferTransition{F,l,tf}, input::AbstractMatrix, ::Type{Val{bsize}} = Val{:auto})

  bcode = (bsize == :auto) ? :(size(input,2)) : :(bsize)

  #check to see if the
  if nounroll(tf)
    quote
      ##########################################################################
      #   NOT UNROLLABLE CASE
      ##########################################################################
      for bdx = 1:$bcode
        output[:, bdx] = tf(output[:, bdx])
      end
      ##########################################################################
    end
  else
    quote
      ##########################################################################
      #   UNROLLABLE CASE
      ##########################################################################
      for bdx = 1:$bcode
        for idx = 1:o
          output[idx, bdx] = tf(output[idx, bdx])
        end
      end
      ##########################################################################
    end
  end
end




include("./backpropagation.jl")
include("./dropout.jl")

end
