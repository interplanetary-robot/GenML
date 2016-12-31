
################################################################################
# back propagation.
hasbackpropagation{F, i, o, tf}(::Type{FullyConnectedLayer{F, i, o, tf}}) = true

#TODO - make all of the i/o value parameters to force recompilation for each
#value.

function reversematrixfma{F}(input_array::AbstractVector{F}, matrix::AbstractMatrix{F}, output_array::AbstractVector{F}, i, o)
  for idx = 1:i
      input_array[idx] = zero(F)
    for jdx = 1:o
      input_array[idx] += matrix[jdx, idx] * output_array[jdx]
    end
  end
end

function scaledouterproductfma{F}(matrix::AbstractMatrix{F}, output_deltas::AbstractVector{F}, input_array::AbstractVector, alpha::F, i, o)
  for idx = 1:o
    for jdx = 1:i
      matrix[idx,jdx] -=  alpha * (output_deltas[idx] * input_array[jdx]) # + lambda * mlp.transitions[l_idx - 1])
    end
  end
end

function scaledsubtract{F}(target_vector::AbstractVector{F}, value_vector::AbstractVector{F}, alpha::F, o)
  for idx = 1:o
    target_vector[idx] -= alpha * value_vector[idx]
  end
end

function dxasychainrule{F}(outer_differential::AbstractVector{F}, inner_value::AbstractVector{F}, f::Function, o)
  for idx = 1:o
    outer_differential[idx] = outer_differential[idx] .* (dxasy(f))(inner_value[idx])
  end
end

@generated function backpropagate!{F, i, o, tf}(fcl::FullyConnectedLayer{F, i, o, tf},
#==#                                            input::AbstractArray,
#==#                                            output::AbstractArray{F},
#==#                                            output_deltas::AbstractArray{F},
#==#                                            input_deltas::VoidableDeltas{F} = nothing)

  code = quote
    #for now.
    const alpha = F(0.1)

    #overwrite the output delta values with the adjusted values taking the
    dxasychainrule(output_deltas, output, tf, o)

    scaledsubtract(fcl.bias, output_deltas, alpha, o)

    scaledouterproductfma(fcl.transition, output_deltas, input, alpha, i, o)
  end

  (input_deltas == Void) && return code

  #handle the backpropagating delta data.
  quote
    $code
    reversematrixfma(input_deltas, fcl.transition, output_deltas, i, o)
  end
end
