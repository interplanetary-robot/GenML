#various utilities to help with differentiation.

d(::Function) = throw(MethodError(d, (Function)))
dxasy(::Function) = throw(MethodError(dbyx, (Function)))

#if you pass a function a vector, should it not be unrolled?
nounroll(::Function) = false
