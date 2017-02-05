import ..Math.d

@generated function backpropagationoptimize{F}(mlmodel::MLAlgorithm{F}, input::AbstractArray, answers::AbstractArray, cost::Function)

  #check to make sure the mlmodel has backpropagation
  if !hasbackpropagation(mlmodel)
    return :(throw(ArgumentError("mlmodel $(typeof(mlmodel)) doesn't support backpropagation")))
  end

  #for analysis purposes, allow the backpropagation of a single layer.
  if nolayers(mlmodel)
    quote
      #TODO:  Add support for this.
      throw(ArgumentError("single layer backprop has not been implemented yet."))
    end
  else
    quote
      #warn("memory optimized version of backpropagation has not been implemented yet")

      #generate a storage unit for the mlmodel
      storage = BackpropStorage(mlmodel)
      #generate a results unit for the mlmodel
      res = Vector{F}(outputs(mlmodel))

      evaluate!(res, mlmodel, input, storage)

      res_deltas = (d(cost))(answers, res)

      backpropagate!(mlmodel, input, res, res_deltas, storage)
    end
  end
end
