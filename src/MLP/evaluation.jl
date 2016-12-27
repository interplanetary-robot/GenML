
# MLAlgorithms should implement evaluate() methods.
buffername{F, LD}(::Type{MultilayerPerceptron{F, LD}}, idx, width) = Symbol("__MLP_BUFFER_", replace(string(F), r"{|}", ""), "_", join(LD, "_"), "_", idx, "x", width)

@generated function create_buffer{F, LD, idx, width}(T::Type{MultilayerPerceptron{F, LD}}, ::Type{Val{idx}}, ::Type{Val{width}})
  buffer_id = buffername(MultilayerPerceptron{F, LD}, idx, width)
  length = LD[idx]
  if width == :v
    :(global const $buffer_id = zeros($F, $length))
  else
    :(global const $buffer_id = zeros($F, $length, $width))
  end
end

@generated function evaluate!{F, LD}(output::Vector{F}, mlp::MultilayerPerceptron{F, LD}, input::Vector)
  code = :()
  #Set the initial source to be the input.
  old_buffer_id = :input
  #run a batch evaluate WITH using buffers for intermediate values.
  for idx = 1:(length(LD) - 2)
    #assign the name of the destination temporary buffer.
    new_buffer_id = buffername(mlp, idx + 1, :v)
    #first generate the temporary holding buffers, if necessary.
    isdefined(new_buffer_id) || create_buffer(mlp, Val{idx + 1}, Val{:v})
    #then store this in the actual loop.
    code = quote
      $code
      global $new_buffer_id #bring the buffer_id constant into scope.
      evaluate!($new_buffer_id, mlp.layers[$idx], $old_buffer_id)
    end
    old_buffer_id = new_buffer_id
  end
  #for the last code segment, direct to the output
  code = quote
    $code
    evaluate!(output, mlp.layers[end], $old_buffer_id)
  end
  #output the code.
  code
end

@generated function batch_evaluate!{F, LD, width}(output::Matrix{F}, mlp::MultilayerPerceptron{F, LD}, input::Matrix, ::Type{Val{width}} = Val{0})
  #run a batch_evaluate without using buffers for intermediate values.
  if (width == 0)
    return quote
      old_buffer = input
      for idx = 1:(length(LD) - 2)
        #allocate a new buffer.
        new_buffer = Matrix{F}(LD[idx + 1], size(input, 2))
        batch_evaluate!(new_buffer, mlp.layers[1], old_buffer)
        old_buffer = new_buffer
      end
      batch_evaluate!(output, mlp.layers[end], old_buffer)
    end
  end

  #initialize a blank code segment which we will unroll manually.
  code = :()
  #Set the initial source to be the input.
  old_buffer_id = :input
  #run a batch evaluate WITH using buffers for intermediate values.
  for idx = 1:(length(LD) - 2)
    #assign the name of the destination temporary buffer.
    new_buffer_id = buffername(mlp, idx + 1, width)
    #first generate the temporary holding buffers, if necessary.
    isdefined(new_buffer_id) || create_buffer(mlp, Val{idx + 1}, Val{width})

    code = quote
      $code
      global $new_buffer_id #bring the buffer_id constant into scope.
      batch_evaluate!($new_buffer_id, mlp.layers[$idx], $old_buffer_id, Val{width})
    end
    old_buffer_id = new_buffer_id
  end
  #for the last code segment, direct to the output
  code = quote
    $code
    batch_evaluate!(output, mlp.layers[end], $old_buffer_id, Val{width})
  end
  #output the code.
  code
end
