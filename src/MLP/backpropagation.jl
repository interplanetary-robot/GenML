#by necessity, backpropagation optimization dependent on the data type itself.

#"functional" symbolic differentiation.
import ..d

@generated function retrieve_buffers{F, LD, width}(::Type{MultilayerPerceptron{F, LD}}, ::Type{Val{width}})
  if width == :v
    code = :(b = Array{F,1}[])
  else
    code = :(b = Array{F,2}[])
  end

  for idx = 2:(length(LD) - 1)
    buffer_id = buffername(MultilayerPerceptron{F, LD}, idx, width)
    code = quote
      $code
      push!(b, $buffer_id)
    end
  end

  quote
    $code
    b
  end
end

function backpropagationoptimize{F,LD}(mlp::MultilayerPerceptron{F,LD}, input::Vector, answers::Vector, cost::Function)

  #set some terms
  alpha = 0.1

  #first, do a forward pass.
  res = mlp(input)

  b = retrieve_buffers(MultilayerPerceptron{F,LD}, Val{:v})

  #calculate the costs (deltas) for the final vector.
  delta_layer = (d(cost))(answers, res)

  backpropagate!(mlp, input, res, delta_layer, Val{true})
#=
  #next calculate the matrix deltas and the nextlayer delta
  for l_idx = (length(LD) - 1) : -1 : 1
    #also calculate the transitions update.
    if l_idx == 1
      matrix_prefix_delta = delta_layer .* (dxasy(sigmoid))(layer_answers)
      transitions_delta = matrix_prefix_delta * (input')
      mlp.layers[l_idx].bias -= alpha * matrix_prefix_delta
      mlp.layers[l_idx].transition -=  alpha * transitions_delta
    else
      matrix_prefix_delta = delta_layer .* (dxasy(sigmoid))(layer_answers)
      delta_previous_layer = (mlp.layers[l_idx].transition') * matrix_prefix_delta
      transitions_delta = matrix_prefix_delta * (b[l_idx - 1]')
      mlp.layers[l_idx].bias -= alpha * matrix_prefix_delta
      mlp.layers[l_idx].transition -=  alpha * transitions_delta # + lambda * mlp.transitions[l_idx - 1])
      #and then update the layer (but avoid changing the bias vector)
      layer_answers = b[l_idx - 1]
      delta_layer = delta_previous_layer
    end
  end
=#
end


hasbackpropagation{F, LD}(::Type{MultilayerPerceptron{F,LD}}) = true

function backpropagate!{F, LD, last}(mlp::MultilayerPerceptron{F,LD}, input_values::Vector, output_values::Vector{F}, output_deltas::Vector{F}, ::Type{Val{last}})
  #presume that the forward propagation has been done, and that the relevant
  #buffers have been executed.
  b = retrieve_buffers(MultilayerPerceptron{F,LD}, Val{:v})

  this_layer_deltas = output_deltas
  this_layer_output = output_values

  for l_idx = (length(LD) - 1) : -1 : 2
    this_layer_deltas = backpropagate!(mlp.layers[l_idx], b[l_idx - 1], this_layer_output, this_layer_deltas, Val{false})
    this_layer_output = b[l_idx - 1]
  end

  backpropagate!(mlp.layers[1], input_values, this_layer_output, this_layer_deltas, Val{last})
end
