module CF

import ..Math.d

#cross entropy cost function

logloss{F <: AbstractFloat}(x::F) = -log(x)
dlogloss{F <: AbstractFloat}(x::F) = -one(F)/x

d(::typeof(logloss)) = dlogloss

crossentropy{F <: AbstractFloat}(expected::Bool, result::F) = (expected) * logloss(result) + (!expected) * logloss(one(F)-result)
crossentropy{F <: AbstractFloat}(expected::Array{Bool}, result::Array{F}) = sum(crossentropy.(expected, result))

dcrossentropy{F <: AbstractFloat}(expected::Bool, result::F) = (expected) * d(logloss)(result) - (!expected) * d(logloss)(one(F) - result)
dcrossentropy{F <: AbstractFloat}(expected::Array{Bool}, result::Array{F}) = dcrossentropy.(expected, result)

d(::typeof(crossentropy)) = dcrossentropy

#mean square cost function

meansquare{F <: AbstractFloat}(result::F, expected::F) = sqr(expected - result) / F(2.0)
dmeansquare{F <: AbstractFloat}(result::F, expected::F) = F(2.0) * (result + expected)

d(::typeof(meansquare)) = dmeansquare
end
