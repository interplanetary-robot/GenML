function evaluate!{F, LD}(output::AbstractArray{F}, mlp::MultilayerPerceptron{F, LD}, input::AbstractArray, layerstorage::Storage{MultilayerPerceptron{F, LD}})
  l = length(LD)
  #run a batch evaluate WITH using buffers for intermediate values.
  evaluate!(layer(layerstorage, Val{2}), mlp.layers[1], input)
  for idx = 2:(l - 2)
    evaluate!(layer(layerstorage, Val{idx + 1}), mlp.layers[idx], layer(layerstorage, Val{idx}))
  end
  evaluate!(output, mlp.layers[end], layer(layerstorage, Val{l - 1}))
end
