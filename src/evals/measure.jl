include("maxK.jl")

function compute_precNrec(matX_ground_truth::SparseMatrixCSC, matX_infere::Array{Float64,2}, topK::Array{Int64,1})
    usr_size = size(matX_ground_truth, 1)
    itm_size = size(matX_ground_truth, 2)
    topK_size = length(topK)
    precision = zeros(Float64, usr_size, topK_size)
    recall = zeros(Float64, usr_size, topK_size)

    if size(matX_infere, 1) != usr_size
        return false
    end

    K = maximum(topK)

    mat_loc = zeros(Int64, usr_size, K)
    for u = 1:usr_size
        (res, loc) = maxK(matX_infere[u,:], K)
        mat_loc[u,:] = loc
    end
    index_ij = findall(x->x>0, mat_loc)
    item_id = mat_loc[index_ij]
    usr_id = Array(1:length(item_id))
    topK_rank = Array(1:length(item_id))
    for ii = 1:length(index_ij)
        usr_id[ii] = index_ij[ii][1]
        topK_rank[ii] = index_ij[ii][2]
    end

    for i = 1:topK_size
        win_size = usr_size * topK[i]

        # dim(accurate_mask) = usr_size * itm_size
        accurate_mask = sparse(usr_id[collect(1:win_size)], item_id[collect(1:win_size)], ones(win_size), usr_size, itm_size)
        accurate_mask .*= matX_ground_truth
        is, js, vs = findnz(accurate_mask)
        accurate_mask = sparse(is, js, ones(nnz(accurate_mask)), size(accurate_mask)...)

        num_TP = sum(accurate_mask, dims=2)[:]
        precision[:,i] = num_TP / topK[i]
        recall[:,i] = num_TP ./ sum(matX_ground_truth.>0, dims=2)[:]
    end

    return precision, recall
end

#
#  /// --- Unit test for function: compute_precNrec --- ///
#
# X =  [0. 5 0 4 0 3 0 2 0 1;
#       1  2 3 4 0 0 0 0 0 0]
# XX = [0. 5 4 0 0 3 0 2 0 1;
#       1  0 0 4 0 0 3 0 1 1]
# xxx = sparse(X)
# compute_precNrec(xxx, XX, [3])


function compute_MRR(matX_ground_truth::SparseMatrixCSC, matX_infere::Array{Float64,2})
    MRR = zeros(size(matX_infere, 1))

    # Variable loc wil receive the CartesianIndex
    res, loc = findmax(matX_ground_truth, 2)
    for u = 1:size(matX_infere, 1)
    	MRR[u] = 1 / (sum((matX_infere[u,:] .- matX_infere[loc[u]]) .> 0) + 1)
    end

    return MRR
end

# compute_MRR(xxx, XX)


function compute_nDCG(matX_ground_truth::SparseMatrixCSC, matX_infere::Array{Float64,2}, topK::Array{Int64,1})
    K = maximum(topK)

    mat_loc = zeros(Int64, usr_size, K)
    for u = 1:usr_size
        (res, loc) = maxK(matX_infere[u,:], K)
        mat_loc[u,:] = loc
    end

    nDCG = zeros(size(matX_infere, 1), length(topK))
    for u = 1:size(matX_infere, 1)
        sort_indx_predict = sortperm(res[:,u], rev=true)
        sort_indx_label = sortperm(vec_label[u, :], rev=true)
        sort_val_label = vec_label[u, sort_indx_label]

        vecRel = vec_label[u, mat_loc[sort_indx_predict, u]]

        for i = 1:length(topK)
            k = topK[i]
            DCG = sum(vecRel[1:k] ./ log2.(Array(1:k) .+ 1))
            IDCG = sum(sort_val_label[1:k] ./ log2.(Array(1:k) .+ 1))
            nDCG[u, i] = DCG / IDCG
        end
    end
end


function LogPRPFObjFunc(C::Float64, alpha::Float64, X::SparseMatrixCSC{Float64, Int64}, XX::Array{Float64, 2})
  #
  # Calculate the log likelihood of the objective function of PRPF
  # X: ground truth matrix
  # XX: predicted matrix
  #
  obj = 0;
  for u = 1:size(X, 1)
    (js, vs) = findnz(X[u, :]);
    if length(vs) == 0
      continue;
    end
    vec_matX_u = full(X[u, js]);
    vec_predict_X_u = full(XX[u, js]);

    mat_diff_matX_u = broadcast(-, vec_matX_u', vec_matX_u);
    mat_diff_predict_X_u = broadcast(-, vec_predict_X_u', vec_predict_X_u);

    sigma_mat_diff_predict_X_u = -log.(1 + exp.(-mat_diff_predict_X_u));

    obj += C / length(vec_matX_u) * sum(sigma_mat_diff_predict_X_u .* (mat_diff_matX_u .> 0))

    if isnan(obj)
      print("NaN");
    end
  end

  is, js, vs = findnz(X)
  vecV = findnz(XX .* sparse(is, js, ones(nnz(X)), size(X)...))[3]
  obj -= alpha * sum( log.(1+exp.(-vecV)) )
  #obj -= alpha * sum( log(1+exp(-XX)) .* sparse(findn(X)..., ones(nnz(X)), size(X)...) )

  return obj
end

#
#  /// --- Unit test --- ///
#
# X =  sparse([5. 4 3 0 0 0 0 0;
#              3. 4 5 0 0 0 0 0;
#              0  0 0 3 3 4 0 0;
#              0  0 0 5 4 5 0 0;
#              0  0 0 0 0 0 5 4;
#              0  0 0 0 0 0 3 4])
#
# XX = [4.0  4.0  4.0  0.0  0.0  0.0  0.0  0.0;
#       4.0  4.0  4.0  0.0  0.0  0.0  0.0  0.0;
#       0.0  0.0  0.0  4.0  3.0  5.0  0.0  0.0;
#       0.0  0.0  0.0  4.0  3.0  5.0  0.0  0.0;
#       0.0  0.0  0.0  0.0  0.0  0.0  4.0  4.0;
#       0.0  0.0  0.0  0.0  0.0  0.0  4.0  4.0]
# LogPRPFObjFunc(1., 1000., X, XX)


function DistributionPoissonLogNZ(X::SparseMatrixCSC{Float64, Int64}, XX::Array{Float64, 2})
    #
    # Calculate the log likelihood with the Poisson distribution (X ~ XX)
    # X: ground truth matrix
    # XX: predicted matrix
    #
    l = 0
    cap_x = log.(XX)

    x_X, y_X, v_X = findnz(X)
    for i = 1:length(v_X)
        a = v_X[i, 1] * cap_x[x_X[i,1], y_X[i,1]] - lgamma(v_X[i, 1] + 1)
        l += a;
    end

    l = l - sum(XX)
end

#
#  /// --- Unit test --- ///
#
# DistributionPoissonLogNZ(X, XX)


# XXXX = readdlm("/Users/iankuoli/Downloads/inferX.csv", ',');
#
# X5 = XXXX - XXXX .* sparse(findn(matX_train)..., ones(nnz(matX_train)), size(matX_train)...)
#
# precision, recall = compute_precNrec(matX_test, X5, [1, 2, 3, 5])
# sum(precision, 1)
# sum(precision, 1)/sum(sum(matX_test,2) .> 0)
# sum(recall,1)/50
#
# sum(sum(matX_test,2) .> 0)
# matX_test[7,:]
