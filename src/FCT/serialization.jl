#the calling function is responsible for allocating the proper number of array slots to do this correctly.
function flatten!{F, i, o, tf}(v::Vector{F}, fcl::FullyConnectedTransition{F, i, o, tf}, offset::Integer = 0)

  #first slot in the biases.
  @inbounds for idx = 1:o
    v[idx + offset] = fcl.bias[idx]
  end

  #next slot in the data.
  @inbounds for idx = 1:(i * o)
    v[idx + offset + o] = fcl.transition[idx]
  end
end

function unflatten!{F, i, o, tf}(fcl::FullyConnectedTransition{F, i, o, tf}, v::Vector{F}, offset::Integer = 0)
  #reverse the process
  @inbounds for idx = 1:o
    fcl.bias[idx] = v[idx + offset]
  end
  #next slot in the data.
  @inbounds for idx = 1:(i * o)
    fcl.transition[idx] = v[idx + offset + o]
  end
end
