immutable MLPStorage{F, LayerDescriptor, N} <: Storage{MultilayerPerceptron{F, LayerDescriptor}, N}
  layer_buffer::Vector{F}
end

immutable MLPBackpropStorage{F, LayerDescriptor, N} <: BackpropStorage{MultilayerPerceptron{F, LayerDescriptor}, N}
  layer_buffer::Vector{F}
  backprop_buffer::Vector{F}
end

#implement allocations that allow creation of the default nonparametric allocations.
function (::Type{Storage}){F, LD}(::Type{MultilayerPerceptron{F, LD}}, N = :v)
  layercount = sum(LD[2:end-1]) * ((N == :v) ? 1 : N)
  MLPStorage{F, LD, N}(Vector{F}(layercount))
end
(::Type{Storage}){T <: MultilayerPerceptron}(::T, N = :v) = Storage(T, N)

function (::Type{BackpropStorage}){F, LD}(::Type{MultilayerPerceptron{F, LD}}, N = :v)
  layercount = sum(LD[2:end-1]) * ((N == :v) ? 1 : N)
  MLPBackpropStorage{F, LD, N}(Vector{F}(layercount), Vector{F}(layercount))
end
(::Type{BackpropStorage}){T <: MultilayerPerceptron}(::T, N = :v) = BackpropStorage(T)

doc"""
  `GenML.MLP.layer(::Storage{MultilayerPerceptron}, Val{N})`

  finds a reference to a layer in the execution storage for an MLP
"""
@generated function layer{F, LD, N, L}(s::Storage{MultilayerPerceptron{F, LD}, N}, ::Type{Val{L}})
  #retrieves the view on the array which represents layer n.  Note:
  #layer 1 is NOT allowed, since that should generically be the "input"
  #layer.
  (L < 2) && throw(ArgumentError("no storage for layers lower than 2"))
  #calculate the batch width
  n = (N == :v) ? 1 : N
  if (L == 2)
    start = 1
    finish = LD[2] * n
  else
    start = sum(LD[2:L-1]) * n + 1
    finish = sum(LD[2:L]) * n
  end

  if N == :v
    :(view(s.layer_buffer, $start:$finish))
  else
    layersize = LD[L]
    :(reshape(view(s.layer_buffer, $start:$finish), $layersize, N))
  end
end

doc"""
  `GenML.MLP.layer(::Storage{MultilayerPerceptron}, Val{N})`

  finds a reference to backpropagation data in the execution storage for an MLP
"""
@generated function backprop_layer{F, LD, N, L}(s::MLPBackpropStorage{F, LD, N}, ::Type{Val{L}})
  #retrieves the view on the array which represents layer n.  Note:
  #layer 1 is NOT allowed, since that should generically be the "input"
  #layer.
  (L < 2) && throw(ArgumentError("no storage for layers lower than 2"))
  if (L == 2)
    start = 1
    finish = LD[2] * n
  else
    start = sum(LD[2:L-1]) * n + 1
    finish = sum(LD[2:L]) * n
  end

  if N == :v
    :(view(s.backprop_buffer, $start:$finish))
  else
    layersize = LD[L]
    :(reshape(view(s.backprop_buffer, $start:$finish), layersize, N))
  end
end
