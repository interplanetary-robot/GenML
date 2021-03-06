module TF

import ..d; import ..dxasy

sigmoid{F <: AbstractFloat}(x::F) = one(F)/(one(F) + exp(-x))
dasysigmoid{F <: AbstractFloat}(y::F) = y * (one(F) - y)
#explicitly reinterpret the dasysigmoid.
dsigmoid{F <: AbstractFloat}(x::F) = dasysigmoid(sigmoid(x))

id{F <: AbstractFloat}(x::F) = x

softplus{F <: AbstractFloat}(x::F) = log(one(F) + exp(x))
dasysoftplus{F <: AbstractFloat}(y::F) = (exp(y) - one(F)) / exp(y)

#broadcast functions over arrays.
sigmoid{F <: AbstractFloat}(a::Array{F})      = sigmoid.(a)
dasysigmoid{F <: AbstractFloat}(a::Array{F})  = dasysigmoid.(a)
dsigmoid{F <: AbstractFloat}(a::Array{F})     = dsigmoid.(a)
id{F <: AbstractFloat}(a::Array{F})           = a
softplus{F <: AbstractFloat}(a::Array{F})     = softplus.(a)
dasysoftplus{F <: AbstractFloat}(a::Array{F}) = dasysoftplus.(a)

#non-simple transfer functions.
function softmax{F <: AbstractFloat}(a::Array{F})
  #go over the entire array and add up the values.
  totalexpsum = zero(F)
  for idx = 1:length(a)
    totalexpsum += exp(a[idx])
  end
  exp.(a) / totalexpsum
end
dsoftmax{F <: AbstractFloat}(a::Array{F}) = dasysigmoid.(softmax(a))

dxasy(::typeof(sigmoid)) = dasysigmoid
dxasy(::typeof(id)) = one
dxasy(::typeof(softplus)) = dasysoftplus
dxasy(::typeof(softmax)) = dasysigmoid

d(::typeof(sigmoid)) = dsigmoid
d(::typeof(id)) = one
d(::typeof(softplus)) = sigmoid
d(::typeof(softmax)) = dsoftmax

end
