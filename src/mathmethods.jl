module Math

doc"""
  `GenML.Math.matrixfma(v_out, M::Matrix, v_bias, v_in [Val{outsize}, Val{insize}, Val{bsize}])`

  does a fused-multply-add of the matrix times a vector or matrix.
  Mathematically the following calculation is run:

    v_out = M * v_in + v_bias                  if passed vectors in the first and fourth positions
    m_out = M * m_in + v_bias ⊗ ones(outsize)  if passed matrices in the first and fourth positions

  where * is standard matrix multiplication and ⊗ is outer product.

  outsize should be the length of the output vector, insize should be the length
  of the input vector, and bsize should be batch size (width of the input matrix).
  The batch size is ignored if the input is a vector.  If the final parameters
  are omitted, they will be autocalculated by measuring the size of the matrix.
"""
@generated function matrixfma{F, osize, isize, bsize}(v_out::AbstractVector{F}, mtx::AbstractMatrix{F}, v_in::AbstractVector, bias::AbstractVector{F},
                                        ::Type{Val{osize}} = Val{:auto}, ::Type{Val{isize}} = Val{:auto}, ::Type{Val{bsize}} = Val{:auto})
  #create initialization code.
  initcode = :()
  (osize == :auto) && initcode = :(osize = size(mtx, 1))
  (isize == :auto) && initcode = :($initcode; isize = size(mtx, 2))
  #batch size is ignored for the vector situation.
  quote
    $initcode
    for idx = 1:osize
      @inbounds v_out[idx] = bias[idx]
      @inbounds for jdx = 1:isize
        v_out[idx] += mtx[idx, jdx] * v_in[jdx]
      end
    end
  end
end
@generated function matrixfma{F, osize, isize, bsize}(m_out::AbstractMatrix{F}, mtx::AbstractMatrix{F}, m_in::AbstractMatrix{F}, bias::AbstractVector{F},
                                        ::Type{Val{osize}} = Val{:auto}, ::Type{Val{isize}} = Val{:auto}, ::Type{Val{bsize}} = Val{:auto})
  #create initialization code.
  initcode = :()
  (osize == :auto) && initcode = :(osize = size(mtx, 1))
  (isize == :auto) && initcode = :($initcode; isize = size(mtx, 2))
  (bsize == :auto) && initcode = :($initcode; bsize = size(m_in, 2))
  quote
    $initcode
    for bdx = 1:bsize
      for idx = 1:osize
        @inbounds output[idx, bdx] = bias[idx]
        @inbounds for jdx = 1:isize
          m_out[idx, bdx] += mtx[idx, jdx] * m_in[jdx, bdx]
        end
      end
    end
  end
end

doc"""
  `GenML.Math.reversematrixmul(v_in, M::Matrix, v_out, [Val{insize}, Val{outsize}, Val{bsize}])`

  does a fused-multply-add of the matrix transpose times a vector or matrix.
  Mathematically the following calculation is run:

    v_in = M' * v_out       (if passed vectors in the first and third position)
    m_in = M' * m_out       (if passed matrices in the first and third position)

  where * is standard matrix multiplication and ' is matrix transposition.

  *Please note the specific definition of in/out parameters and the reversal of
  the Value parameters relative to `GenML.Math.matrixfma`*

  outsize should be the length of the output vector, insize should be the length
  of the input vector, and bsize should be batch size (width of the output matrix).
  The batch size is ignored if the input is a vector.  If the final parameters
  are omitted, they will be autocalculated by measuring the size of the matrix.
"""
@generated function reversematrixmul{F, isize, osize, bsize}(v_in::AbstractVector{F}, matrix::AbstractMatrix{F}, v_out::AbstractVector{F},
                                                  ::Type{Val{isize}} = Val{:auto}, ::Type{Val{osize}} = Val{:auto}, ::Type{Val{bsize}} = Val{:auto})
  #create initialization code.
  initcode = :()
  (osize == :auto) && initcode = :(osize = size(mtx, 1))
  (isize == :auto) && initcode = :($initcode; isize = size(mtx, 2))
  #batch size is ignored for the vector situation.
  quote
    $initcode
    for idx = 1:isize
      @inbounds v_in[idx] = zero(F)
      for jdx = 1:osize
        @inbounds v_in[idx] += matrix[jdx, idx] * v_out[jdx]
      end
    end
  end
end
@generated function reversematrixmul{F, isize, osize, bsize}(m_in::AbstractVector{F}, matrix::AbstractMatrix{F}, m_out::AbstractVector{F},
                                                  ::Type{Val{isize}} = Val{:auto}, ::Type{Val{osize}} = Val{:auto}, ::Type{Val{bsize}} = Val{:auto})
  #create initialization code.
  initcode = :()
  (osize == :auto) && initcode = :(osize = size(mtx, 1))
  (isize == :auto) && initcode = :($initcode; isize = size(mtx, 2))
  (bsize == :auto) && initcode = :($initcode; bsize = size(m_out, 2))
  #batch size is ignored for the vector situation.
  quote
    $initcode
    for bdx = 1:bsize
      for idx = 1:isize
        @inbounds m_in[idx, bdx] = zero(F)
        for jdx = 1:osize
          @inbounds m_in[idx, bdx] += matrix[jdx, idx] * m_out[jdx, bdx]
        end
      end
    end
  end
end

doc"""
  `GenML.Math.scaledouterproductfms(M::Matrix, d_out, v_in, alpha, [Val{outsize}, Val{insize}, Val{bsize}])`

  does a scaled outer product fused multiply subtract of the matrix transpose
  times a vector or matrix.  Mathematically the following calculation is run:

    M = M - alpha * d_out ⊗ v_in                      (if passed vectors in the second and third position)
    M = M - ∑(k) alpha * d_out[:,k] ⊗ m_in[:,k]       (if passed matrices in the second and third position)

  where * is standard matrix multiplication and ⊗ is vector outer product.

  *Please note the specific definition of the parameters and the ordering of
  the Value parameters*

  outsize should be the length of the output vector, insize should be the length
  of the input vector, and bsize should be batch size (width of the output matrix).
  The batch size is ignored if the input is a vector.  If the final parameters
  are omitted, they will be autocalculated by measuring the size of the matrix.
"""
@generated function scaledouterproductfms{F, osize, isize, bsize}(matrix::AbstractMatrix{F}, d_out::AbstractVector{F}, d_in::AbstractVector, alpha::F,
                              ::Type{Val{osize}} = Val{:auto}, ::Type{Val{isize}} = Val{:auto}, ::Type{Val{bsize}} = Val{:auto})
  #create initialization code.
  initcode = :()
  (osize == :auto) && initcode = :(osize = size(mtx, 1))
  (isize == :auto) && initcode = :($initcode; isize = size(mtx, 2))
  #batch size is ignored for the vector situation.
  quote
    $initcode
    for idx = 1:osize
      for jdx = 1:isize
        matrix[idx,jdx] -=  alpha * (d_out[idx] * d_in[jdx])
      end
    end
  end
end
@generated function scaledouterproductfms{F, osize, isize, bsize}(matrix::AbstractMatrix{F}, d_out::AbstractMatrix{F}, v_in::AbstractMatrix, alpha::F,
                              ::Type{Val{osize}} = Val{:auto}, ::Type{Val{isize}} = Val{:auto}, ::Type{Val{bsize}} = Val{:auto})
  #create initialization code.
  initcode = :()
  (osize == :auto) && initcode = :(osize = size(mtx, 1))
  (isize == :auto) && initcode = :($initcode; isize = size(mtx, 2))
  (bsize == :auto) && initcode = :($initcode; bsize = size(d_out, 2))
  #batch size is ignored for the vector situation.
  quote
    $initcode
    for idx = 1:osize
      for jdx = 1:isize
        accumulator = zero(F)
        for bdx = 1:bsize
          accumulator += d_out[idx, bdx] * v_in[jdx, bdx]
        end
        matrix[idx,jdx] -= alpha * accumulator
      end
    end
  end
end

doc"""
  `GenML.Math.scaledsubtract(v::Vector, values, alpha, [Val{vsize}, Val{bsize}])`

  does a scaled subtraction.  Mathematically the following calculation is run:

    v = v - alpha * values                     (if passed a vector in the second position)
    v = v - ∑(k) alpha * values[:,k]           (if passed a matrix in the second position)

  v should be the length of the target (and values) vector, and bsize should be
  batch size (width of the values matrix). The batch size is ignored if the
  input is a vector.  If the final parameters are omitted, they will be
  autocalculated by measuring the size of the target vector.
"""
function scaledsubtract{F, vsize, bsize}(target_vector::AbstractVector{F}, value_vector::AbstractVector{F}, alpha::F,
                                         ::Type{Val{vsize}} = Val{auto}, ::Type{Val{bsize}} = Val{auto})
  #create initialization code.
  initcode = :()
  (vsize == :auto) && initcode = :(vsize = length(target_vector))
  quote
    $initcode
    for idx = 1:vsize
      target_vector[idx] -= alpha * value_vector[idx]
    end
  end
end
function scaledsubtract{F, vsize, bsize}(target_vector::AbstractMatrix{F}, value_vector::AbstractMatrix{F}, alpha::F,
                                         ::Type{Val{vsize}} = Val{auto}, ::Type{Val{bsize}} = Val{auto})
  #create initialization code.
  initcode = :()
  (vsize == :auto) && initcode = :(vsize = size(target_vector, 1))
  (bsize == :auto) && initcode = :(initcode; bsize = size(value_vector, 2))
  quote
    $initcode
    for idx = 1:vsize
      accumulator = zero(F)
      for bdx = 1:bsize
        accumulator += value_vector[idx, bdx]
      end
      target_vector[idx] -= alpha * accumulator
    end
  end
end

doc"""
  `GenML.Math.dxasychainrule(diff::Vector, value::Vector, f::Function, [Val{vsize}, Val{bsize}])`

  A special differential chain rule engine.  Performs the following operation:

  diff = diff .* D(f)(value)

  where D is the differential operator for a function f which takes the *y*
  values and releases d/dx of function f at the points corresponding to
  to those y values.  vsize should be the length of the derivative vector, and
  bsize should be batch size (width of the values matrix). The batch size is
  ignored if the input is a vector.  If the final parameters are omitted,
  they will be autocalculated by measuring the size of the target vector.
"""
function dxasychainrule{F, vsize, bsize}(outer_differential::AbstractVector{F}, inner_value::AbstractVector{F}, f::Function,
                        ::Type{Val{vsize}} = Val{:auto}, ::Type{Val{bsize}} = Val{:auto})
  #create initialization code.
  initcode = :()
  (vsize == :auto) && initcode = :(vsize = length(outer_differential))
  if nounroll(dxasy(f))
    quote
      $initcode
      outer_differential .*= (dxasy(f))(inner_value)
    end
  else
    quote
      $initcode
      for idx = 1:vsize
        outer_differential[idx] *= (dxasy(f))(inner_value[idx])
      end
    end
  end
end
function dxasychainrule{F, vsize, bsize}(outer_differential::AbstractMatrix{F}, inner_value::AbstractMatrix{F}, f::Function,
                        ::Type{Val{vsize}} = Val{:auto}, ::Type{Val{bsize}} = Val{:auto})
  #create initialization code.
  initcode = :()
  (vsize == :auto) && initcode = :(vsize = size(outer_differential, 1))
  (bsize == :auto) && initcode = :(initcode; bsize = size(outer_differential, 2))
  if nounroll(dxasy(f))
    quote
      $initcode
      for bdx = 1:bsize
        outer_differential[:,bdx] = (dxasy(f))(inner_value[:,bdx])
      end
    end
  else
    quote
      $initcode
      for bdx = 1:bsize
        for idx = 1:vsize
          outer_differential[idx,bdx] = outer_differential[idx,bdx] .* (dxasy(f))(inner_value[idx,bdx])
        end
      end
    end
  end
end

end  #modlue
