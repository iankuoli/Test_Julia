
function Learn_ExpoMF(X::modelExpoMF, is_X_train::Array{Int,1}, js_X_train::Array{Int,1}, vs_X_train::Array{Float64,1})

    ############################################################################
    # Update matTheta & matBeta
    # -------------------------
    # E-step
    vec_prob = sqrt.(X.valLambda_y / (2 * pi)) * exp.(-X.valLambda_y * (X.matTheta[usr_idx,:] * X.matBeta[itm_idx,:]') .^ 2 / 2)
    vec_matA_vs = broadcast(*, vec_prob, X.vecMu[itm_idx]')
    vec_matA_vs = vec_matA_vs ./ broadcast(+, vec_matA_vs, (1 .- X.vecMu[itm_idx])')
    is_tmp, js_tmp, vs_tmp = findnz(matX_train[usr_idx, itm_idx])
    matIdty = sparse(is_tmp, js_tmp, ones(Float64,length(js_tmp)), usr_idx_len, itm_idx_len)
    vec_matA_vs = vec_matA_vs - vec_matA_vs .* matIdty + matIdty
    if isnan(sum(vec_matA_vs)) == true
        print("NaN\n")
    end

    # M-step
    tmp_py = vec_matA_vs .* matX_train[usr_idx, itm_idx]
    tmp_theta_mean = tmp_py * X.matBeta[itm_idx, :]
    for u = 1:usr_idx_len
        tmp_matrix = X.valLambda_y * broadcast(*, X.matBeta[itm_idx,:]', vec_matA_vs[u,:]') * X.matBeta[itm_idx,:] +
                     X.valLambda_theta * sparse(1:K, 1:K, ones(Float64, K), K, K)
        X.matTheta[usr_idx[u],:] = X.valLambda_y * tmp_theta_mean[u,:]' * tmp_matrix.^-1
    end
    tmp_beta_mean = tmp_py' * X.matTheta[usr_idx,:];
    for i = 1:itm_idx_len
        tmp_matrix = X.valLambda_y * X.matTheta[usr_idx,:]' * broadcast(*, X.matTheta[usr_idx,:], vec_matA_vs[:,i]) +
                     X.valLambda_beta * sparse(1:K, 1:K, ones(Float64, K), K, K)
        X.matBeta[itm_idx[i],:] = X.valLambda_y * tmp_beta_mean[i,:]' * tmp_matrix.^-1
    end

    # Update priors \mu_i
    X.vecMu[itm_idx] = (X.val_alpha1 .+ sum(vec_matA_vs, dims=1)' .- 1) / (X.val_alpha1 + X.val_alpha2 + usr_idx_len - 2)


end
