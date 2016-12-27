################################################################################
# implementation of the GenML interface

inputs{F, LD}(::Type{MultilayerPerceptron{F, LD}}) = first(LD)
outputs{F, LD}(::Type{MultilayerPerceptron{F, LD}}) = last(LD)

@generated function parameters{F, LD}(::Type{MultilayerPerceptron{F, LD}})
  totalparams = 0
  for idx = 1:(length(LD) - 1)
    #just use a dummy function here, since it doesn't affect the parameter count.
    totalparams += parameters(FullyConnectedLayer{F, LD[idx], LD[idx + 1], TF.sigmoid})
  end
  :($totalparams)
end

@generated function flatten!{F, LD}(v::Vector{F}, mlp::MultilayerPerceptron{F, LD}, offset::Integer = 0)
  l = length(LD)
  #initialize various parameters.
  current_offset = 0
  code = :()
  #manually unroll this.
  for idx = 1:(length(LD) - 1)
    code = quote
      $code
      flatten!(v, mlp.layers[$idx], $current_offset)
    end
    current_offset += parameters(FullyConnectedLayer{F, LD[idx], LD[idx + 1], TF.sigmoid})
  end
  code
end

@generated function unflatten!{F, LD}(mlp::MultilayerPerceptron{F, LD}, storage::Vector{F}, offset::Integer = 0)
  l = length(LD)
  #initialize various parameters.
  current_offset = 0
  code = :( )
  #manually unroll this.
  for idx = 1:(length(LD) - 1)
    code = quote
      $code
      unflatten!(mlp.layers[$idx], storage, $current_offset)
    end
    current_offset += parameters(FullyConnectedLayer{F, LD[idx], LD[idx + 1], TF.sigmoid})
  end
  code
end
