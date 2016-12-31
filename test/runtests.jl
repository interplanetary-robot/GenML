using GenML
using Base.Test

include("xortest.jl")

#include("xortraintest-1.jl") #noisy xor training test
#include("xortraintest-2.jl") #unreliable xor training test

#include("xortraintest-3.jl") #xor training test using PSO - currently not hyperparameter-optimized.
include("xortraintest-4.jl") #xor training test using backpropagation
