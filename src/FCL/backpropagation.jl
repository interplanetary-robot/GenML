
################################################################################
# back propagation.
hasbackpropagation{F, i, o, tf}(::Type{FullyConnectedLayer{F, i, o, tf}}) = true

#TODO - make all of the i/o value parameters to force recompilation for each
#value.

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
