import GenML.Math.matrixfma

function evaluate!{F, i, o, tf}(output::AbstractArray{F}, fcl::FullyConnectedLayer{F, i, o, tf}, input::AbstractArray)
  matrixfma(output, fcl.transition, input, fcl.bias, Val{i}, Val{o})
  for idx = 1:o
    output[idx] = tf(output[idx])
  end
end
