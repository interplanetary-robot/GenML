#math.jl - agglomeration of mathematics files.

module Math
  #general stuff

  #if you pass a function a vector, should it not be unrolled?
  nounroll(::Function) = false

  #various utilities to help with differentiation.
  d(::Function) = throw(MethodError(d, (Function)))
  dxasy(::Function) = throw(MethodError(dbyx, (Function)))

  include("linalg.jl")
end #module math

#cost functions and transfer functions live in their own submodule.

include("costfunctions.jl")
include("transferfunctions.jl")

function Base.convert{F <: AbstractFloat}(::Type{Bool}, pval::F)
  (pval < zero(F)) && throw(InexactError())
  (pval > one(F)) && throw(InexactError())
  return pval > F(0.5)
end
