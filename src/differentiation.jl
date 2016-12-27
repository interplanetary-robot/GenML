#various utilities to help with differentiation.

d(::Function) = throw(MethodError(d, (Function)))
dxasy(::Function) = throw(MethodError(dbyx, (Function)))
