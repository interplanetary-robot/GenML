function evaluate!{F, LD, bsize}(output::AbstractArray{F}, mlp::MultilayerPerceptron{F, LD}, input::AbstractArray, layerstorage::Storage{MultilayerPerceptron{F, LD}}, ::Type{Val{bsize}} = Val{:auto})
  l = length(LD)
  #run a batch evaluate WITH using buffers for intermediate values.
  evaluate!(layer(layerstorage, Val{2}), mlp.layers[1], input, Val{bsize})
  for idx = 2:(l - 2)
    evaluate!(layer(layerstorage, Val{idx + 1}), mlp.layers[idx], layer(layerstorage, Val{idx}), Val{bsize})
  end
  evaluate!(output, mlp.layers[end], layer(layerstorage, Val{l - 1}), Val{bsize})
end
