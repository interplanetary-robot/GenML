hasbackpropagation{F, LD}(::Type{MultilayerPerceptron{F,LD}}) = true

#implementation of the backpropagation algorithm for multilayer perceptrons.
function backpropagate!{F, LD}(mlp::MultilayerPerceptron{F, LD},
                               input::AbstractArray,
                               output::AbstractArray{F},
                               output_deltas::AbstractArray{F},
                               backpropstorage::BackpropStorage{MultilayerPerceptron{F, LD}},
                               input_deltas::VoidableDeltas{F} = nothing)

  l = length(LD) - 1
  #presume that the forward propagation has been done, and that the relevant
  #buffers have been executed.

  this_layer_deltas = output_deltas
  this_layer_output = output

  for l_idx = l:-1:2  #track back through the array.
    this_layer_input_ref = layer(backpropstorage, Val{l_idx})
    next_layer_delta_ref = backprop_layer(backpropstorage, Val{l_idx})

    #call backpropagation on the sublayer, which should be fully connected.
    backpropagate!(mlp.layers[l_idx], this_layer_input_ref, this_layer_output, this_layer_deltas, next_layer_delta_ref)

    #move the references over.
    this_layer_deltas = next_layer_delta_ref
    this_layer_output = this_layer_input_ref
  end

  #solve the last layer, and if the backpropagation isn't supposed to terminate here, pass on the next array.
  backpropagate!(mlp.layers[1], input, this_layer_output, this_layer_deltas, input_deltas)
end
