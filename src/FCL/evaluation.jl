import GenML.Math.matrixfma
import GenML.Math.nounroll

@generated function evaluate!{F, i, o, tf, bsize}(output::AbstractVector{F},
  fcl::FullyConnectedLayer{F, i, o, tf}, input::AbstractVector, ::Type{Val{bsize}} = Val{:auto})

  (bsize != :v) && (bsize != :auto) && return :(throw(ArgumentError("vector evaluation can't do a numerical bsize.")))

  if nounroll(tf)
    transfer_code = :(output[:] = tf(output))
  else
    transfer_code = quote
      for idx = 1:o
        output[idx] = tf(output[idx])
      end
    end
  end

  quote
    matrixfma(output, fcl.transition, input, fcl.bias, Val{o}, Val{i}, Val{bsize})

    $transfer_code
  end
end

@generated function evaluate!{F, i, o, tf, bsize}(output::AbstractMatrix{F},
  fcl::FullyConnectedLayer{F, i, o, tf}, input::AbstractMatrix, ::Type{Val{bsize}} = Val{:auto})

  (bsize != :v) && (bsize != :auto) && return :(throw(ArgumentError("vector evaluation can't do a numerical bsize.")))

  bcode = (bsize == :auto) ? :(size(input,2)) : :(bsize)

  if nounroll(tf)
    transfer_code = quote
      for bdx = 1:$bcode
        output[:, bdx] = tf(output[:, bdx])
      end
    end
  else
    transfer_code = quote
      for bdx = 1:$bcode
        for idx = 1:o
          output[idx, bdx] = tf(output[idx, bdx])
        end
      end
    end
  end

  quote
    matrixfma(output, fcl.transition, input, fcl.bias, Val{o}, Val{i}, Val{bsize})

    $transfer_code
  end
end
