
function matrixfma(output, mtx, input::AbstractVector, bias, isize, osize)
  for idx = 1:osize
    @inbounds output[idx] = bias[idx]
    @inbounds for jdx = 1:isize
      output[idx] += mtx[idx, jdx] * input[jdx]
    end
  end
end

function matrixfma(output, mtx, input::AbstractMatrix, bias, isize, osize)
  for mdx = 1:size(input, 2)
    for idx = 1:osize
      @inbounds output[idx, mdx] = bias[idx]
      @inbounds for jdx = 1:isize
        output[idx, mdx] += mtx[idx, jdx] * input[jdx, mdx]
      end
    end
  end
end

function evaluate!{F, i, o, tf}(output::AbstractArray{F}, fcl::FullyConnectedLayer{F, i, o, tf}, input::AbstractArray)
  matrixfma(output, fcl.transition, input, fcl.bias, i, o)
  for idx = 1:o
    output[idx] = tf(output[idx])
  end
end
