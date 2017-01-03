#math.jl - agglomeration of mathematics files.

module Math
  #general stuff

  #if you pass a function a vector, should it not be unrolled?
  nounroll{F <: Function}(::F) = false

  #various utilities to help with differentiation.
  d(::Function) = throw(MethodError(d, (Function)))
  dxasy(::Function) = throw(MethodError(dbyx, (Function)))

  include("linalg.jl")
end #module math

#cost functions and transfer functions live in their own submodule.

include("costfunctions.jl")
include("transferfunctions.jl")
