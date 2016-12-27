
################################################################################
# back propagation.
hasbackpropagation{F, i, o, tf}(::Type{FullyConnectedLayer{F, i, o, tf}}) = true

@generated function backpropagate!{F, i, o, tf, last}(fcl::FullyConnectedLayer{F, i, o, tf}, input_values::Vector, output_values::Vector{F}, output_deltas::Vector{F}, ::Type{Val{last}})
  code = quote
    #for now.
    const alpha = F(0.1)

    #println("iv:", input_values)
    #println("ov:", output_values)
    #println("od:", output_deltas)

    #calculate the delta for BEFORE the transfer function.
    pretransfer_delta = output_deltas .* (dxasy(tf))(output_values)

    fcl.bias -= alpha * pretransfer_delta
    for idx = 1:o
      for jdx = 1:i
        fcl.transition[idx,jdx] -=  alpha * (pretransfer_delta[idx] * input_values[jdx]) # + lambda * mlp.transitions[l_idx - 1])
      end
    end
  end

  last && return code

  #handle the backpropagating delta data.
  quote
    $code
    #for now, instantiate this here.  We'll preallocate this elsewhere.
    backpropagating_delta = Array{F,1}(i)
    #return the delta to be backpropagated to the next layer.
    for idx = 1:i
        backpropagating_delta[idx] = zero(F)
      for jdx = 1:o
        backpropagating_delta[idx] += fcl.transition[jdx, idx] * pretransfer_delta[jdx]
      end
    end
    backpropagating_delta
    #exit()
  end
end
