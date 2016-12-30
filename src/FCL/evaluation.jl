
function matrixfma(output, mtx, input, bias, isize, osize)
  for idx = 1:osize
    @inbounds output[idx] = bias[idx]
    @inbounds for jdx = 1:isize
      output[idx] += mtx[idx, jdx] * input[jdx]
    end
  end
end

function matrixfma(output, mtx, input, bias, isize, osize, N)
  for mdx = 1:N
    for idx = 1:osize
      @inbounds output[idx, mdx] = bias[idx]
      @inbounds for jdx = 1:isize
        output[idx, mdx] += mtx[idx, jdx] * input[jdx, mdx]
      end
    end
  end
end

function evaluate!{F, i, o, tf}(output::Vector{F}, fcl::FullyConnectedLayer{F, i, o, tf}, input::Vector)
  matrixfma(output, fcl.transition, input, fcl.bias, i, o)
  for idx = 1:o
    output[idx] = tf(output[idx])
  end
end

@generated function batch_evaluate!{F, i, o, tf, N}(output::Matrix{F}, fcl::FullyConnectedLayer{F, i, o, tf}, input::Matrix, ::Type{Val{N}} = Val{0})
  if (N == 0)
    quote
      @inbounds output[:] = fcl.transition * input
      for idx = 1:size(output, 2)
        @inbounds output[:, idx] += fcl.bias
      end
      for idx = 1:length(output)
        @inbounds output[idx] = tf(output[idx])
      end
    end
  else
    quote
      matrixfma(output, fcl.transition, input, fcl.bias, i, o, N)
      for idx = 1:length(output)
        @inbounds output[idx] = tf(output[idx])
      end
    end
  end
end
