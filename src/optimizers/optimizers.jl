module Optimizers
  #collection of optimizers.

  import ..MLAlgorithm

  include("gradientdescent.jl")
  include("particleswarm.jl")
  include("backpropagation.jl")

end
