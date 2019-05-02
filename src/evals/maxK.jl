using DataStructures


function maxK(vecInput::Array{Float64,1}, K::Int64)

    tupleInput = collect(zip(vecInput, 1:length(vecInput)))

    h = DataStructures.BinaryMinHeap(tupleInput[1:K])

    for ii = (K+1):length(tupleInput)
        DataStructures.push!(h, tupleInput[ii])
        DataStructures.pop!(h)
    end

    retVal = zeros(Float64, K)
    retIdx = zeros(Int64, K)

    for ii = 1:K
        val, idx = DataStructures.pop!(h)
        retVal[K+1-ii] = val
        retIdx[K+1-ii] = idx
    end

    return retVal, retIdx
end


#
#  /// --- Unit test --- ///
#
#a = [5., 6., 3., 1., 2., 5., 2., 3., 10., 7.]
#maxK(a, 5)
