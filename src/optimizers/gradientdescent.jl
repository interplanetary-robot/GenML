import ..@import_interface
@import_interface

#one step of gradient optimization.  This is done as a closure so that memory
#operations can be conserved.
@generated function gradientoptimize{F}(mlmodel::MLAlgorithm{F}, input::Matrix, answers::Matrix, cost::Function)
  parameter_count = parameters(mlmodel)
  output_count = outputs(mlmodel)
  quote
    #hyperparameters:
    alpha_parameter = 0.01
    delta_x = 0.0001
    #return this as the stepping function.
    #generate a storage array.
    storage_array = Vector{F}($parameter_count)
    delta_array = Vector{F}($parameter_count)
    slope_array = Vector{F}($parameter_count)

    const input_width = size(input, 2)
    result_array = Matrix{F}($output_count, input_width)

    magfactor = alpha_parameter / delta_x

    for round = 1:300
      #flatten the mlmodel into the storage array
      flatten!(storage_array, mlmodel)
      #calculate the base cost.
      batch_evaluate!(result_array, mlmodel, input, Val{input_width})

      base_cost = cost(answers, result_array)

      @inbounds delta_array[:] = storage_array[:]

      @inbounds for idx = 1:$parameter_count
        old_value = delta_array[idx]
        #nudge the delta array over a bit.
        delta_array[idx] += delta_x
        #deserialize the delta array into the model, then execute the model.
        unflatten!(mlmodel, delta_array)

        batch_evaluate!(result_array, mlmodel, input, Val{input_width})
        #calculate the cost of this execution round.
        delta_cost = cost(answers, result_array)
        #take the delta to generate the slopes.
        slope_array[idx] = base_cost - delta_cost
        #restore the old value
        delta_array[idx] = old_value
      end
      #update the storage array.
      storage_array += magfactor * slope_array
      #unflatten the new storage array into the xornet.
      unflatten!(mlmodel, storage_array)
    end
  end
end
