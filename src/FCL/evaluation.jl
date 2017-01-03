import GenML.Math.matrixfma

function evaluate!{F, i, o, tf, bsize}(output::AbstractArray{F},
  fcl::FullyConnectedLayer{F, i, o, tf}, input::AbstractArray, ::Type{Val{bsize}} = Val{:auto})
  matrixfma(output, fcl.transition, input, fcl.bias, Val{o}, Val{i}, Val{bsize})
  for idx = 1:o
    output[idx] = tf(output[idx])
  end
end
