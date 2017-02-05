
import ..FCT.FCT_Dropouts

type MLP_Dropouts{F, LD} <: DropoutStorage{MultilayerPerceptron{F, LD}}
  layerdropouts::Vector{FCT_Dropouts}
end

#FCL's have dropout.
hasdropout{F, LD}(::Type{MultilayerPerceptron{F, LD}}) = true

function generate_dropout_storage{F, LD}(mlp::MultilayerPerceptron{F, LD})
  layerdropouts = Vector{FCT_Dropouts}(length(LD) - 1)
  for idx = 1:(length(LD) - 1)
    layerdropouts[idx] = generate_dropout_storage(mlp.layers[idx])
  end
  MLPDropouts{F,LD}(layerdropouts)
end

function dropout!{F, LD}(l::MultilayerPerceptron{F, LD}, dv::MLP_Dropouts{F, LD})
  for l_idx = 1:(length(LD) - 1)
    dropout!(l.layers[l_idx], dv.layerdropouts[l_idx])
  end
end

function restore!{F, LD}(l::MultilayerPerceptron{F, LD}, dv::MLP_Dropouts{F, LD})
  for l_idx = 1:(length(LD) - 1)
    restore!(l.layers[l_idx], dv.layerdropouts[l_idx])
  end
end
