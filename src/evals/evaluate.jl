include("measure.jl")


function infer_N_eval(matX::SparseMatrixCSC{Float64, Int}, matX_train::SparseMatrixCSC{Float64, Int},
                      matTheta::Array{Float64,2}, matBeta::Array{Float64,2},
                      topK::Array{Int,1}, vec_usr_idx::Array{Int,1}, j::Int, step_size::Int; ndcg::Bool=true)

    range_step = collect((1 + (j-1) * step_size):min(j*step_size, length(vec_usr_idx)))

    # Compute the Precision and Recall
    matPredict = matTheta[vec_usr_idx[range_step],:] * matBeta'
    is, js, vs = findnz(matX_train[vec_usr_idx[range_step], :])
    tmp_mask = sparse(is, js, ones(nnz(matX_train[vec_usr_idx[range_step], :])),
                      size(matX_train[vec_usr_idx[range_step], :])...)

    matPredict -= matPredict .* tmp_mask
    (vec_precision, vec_recall) = compute_precNrec(matX[vec_usr_idx[range_step], :], matPredict, topK)
    vec_MRR = compute_MRR(matX[vec_usr_idx[range_step], :], matPredict)
    vecPrecision = sum(vec_precision, dims=1)[:]
    vecRecall = sum(vec_recall, dims=1)[:]
    valMRR = sum(vec_MRR, dims=1)[:]
    denominator = length(range_step)

    if ndcg == true
        return vcat(vecPrecision, vecRecall, vecNDCG, valMRR, denominator)
    else
        return vcat(vecPrecision, vecRecall, valMRR, denominator)
    end

end


function Evaluate(matX::SparseMatrixCSC{Float64, Int}, matX_train::SparseMatrixCSC{Float64, Int},
                  matTheta::Array{Float64,2}, matBeta::Array{Float64,2}, topK::Array{Int,1}; ndcg::Bool=true)

    vec_usr_idx = findall(x->x>0, sum(matX, dims=2)[:])
    step_size = 300

    ret_tmp = @distributed (+) for j = 1:Int(ceil(size(matX)[1]/step_size))
        infer_N_eval(matX, matX_train, matTheta, matBeta, topK, vec_usr_idx, j, step_size, ndcg=ndcg)
    end
    ret_tmp /= ret_tmp[end]

    if ndcg == true
        dictResult = Dict("Precision" => ret_tmp[1:length(topK)],
                          "Recall" => ret_tmp[(length(topK)+1):2*length(topK)],
                          "nDCG" => ret_tmp[(length(topK)*2+1):3*length(topK)],
                          "MRR" => ret_tmp[end-1])
    else
        dictResult = Dict("Precision" => ret_tmp[1:length(topK)],
                          "Recall" => ret_tmp[(length(topK)+1):2*length(topK)],
                          "MRR" => ret_tmp[end-1])
    end

    return dictResult
end


#  /// --- Unit test for function: evaluate() --- ///
#
# X =  sparse([5. 4 3 0 0 0 0 0;
#              3. 4 5 0 0 0 0 0;
#              0  0 0 3 3 4 0 0;
#              0  0 0 5 4 5 0 0;
#              0  0 0 0 0 0 5 4;
#              0  0 0 0 0 0 3 4])
# A = [1. 0 0; 1 0 0; 0 1 0; 0 1 0; 0 0 1; 0 0 1]
# B = [4. 0 0; 4 0 0; 4 0 0; 0 4 0; 0 3 0; 0 5 0; 0 0 4; 0 0 4]
# XX = spzeros(6,8)
# topK = [1, 2]
# C = 1.
# alpha = 1000.
# prec, rec, log_likelihood = Evaluate(X, XX, A, B, topK,ndcg=false)
#
# matX = X
# matX_train = XX
# matTheta = A
# matBeta = B
# vec_usr_idx = findall(x->x>0, sum(matX, dims=2)[:])
# step_size = 300
# infer_N_eval(matX, matX_train, matTheta, matBeta, topK, vec_usr_idx, 1, step_size, ndcg=false)


# theta1 = readdlm("/Users/iankuoli/Downloads/theta1.csv", ',')
# beta1 = readdlm("/Users/iankuoli/Downloads/beta1.csv", ',')
# precision, recall, likelihood = evaluate(matX_test, matX_train, theta1, beta1, topK, C, alpha)
#
# if precision == 0.629787
#   println("right")
# end


# using HDF5
#
# beta1 = h5read("/Users/iankuoli/Downloads/beta1.h5", "/dataset1")
# theta1 = h5read("/Users/iankuoli/Downloads/theta1.h5", "/dataset1")
#
# precision, recall, likelihood = evaluate(matX_test, matX_train, theta1, beta1, topK, C, alpha)
#
