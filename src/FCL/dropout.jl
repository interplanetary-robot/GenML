
type FCLDropouts{F, i, o, tf} <: DropoutStorage{FullyConnectedLayer{F, i, o, tf}}
  droppedbiases::Vector{F}
  droppedtransitions::Matrix{F}
end

#FCL's have dropout.
hasdropout{F, i, o, tf}(::Type{FullyConnectedLayer{F, i, o, tf}}) = true

function generate_dropout_storage{F, i, o, tf}(fcl::FullyConnectedLayer{F, i, o, tf})
  FCLDropouts{F, i, o, tf}(zeros(F, o), zeros(F, o, i))
end

function dropout!{F, i, o, tf}(l::FullyConnectedLayer{F, i, o, tf}, dv::FCLDropouts{F, i, o, tf})
  for o_idx = 1:o
    @inbounds dv.droppedbiases[o_idx] = rand(Bool) * l.bias[o_idx]
    @inbounds l.bias[o_idx] -= dv.droppedbiases[o_idx]
  end

  for i_idx = 1:i
    for o_idx = 1:o
      @inbounds dv.droppedtransitions[o_idx, i_idx]  = (l.bias[o_idx] != 0) * l.transition[o_idx, i_idx]
      @inbounds l.transition[o_idx, i_idx]          -= dv.droppedtransitions[o_idx,i_idx]
    end
  end
end

function restore!{F, i, o, tf}(l::FullyConnectedLayer{F, i, o, tf}, dv::FCLDropouts{F, i, o, tf})
  for o_idx = 1:o
    @inbounds l.bias[o_idx] += dv.droppedbiases[o_idx]
  end

  for i_idx = 1:i
    for o_idx = 1:o
      @inbounds l.transition[o_idx, i_idx] += dv.droppedtransitions[o_idx,i_idx]
    end
  end
end
