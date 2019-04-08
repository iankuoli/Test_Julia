#
# Learning phase of Generator
# The Parameters of generator are updated by (stochastic) variaitonal inference
# J = -(D(G(z)) - 1)^2 + p(\omage_d \vert \alpha_d)
# size(matSamples) = (usr_idx, itm_idx * R), where R is the number of samples per entry
#
function Learn_HPF(X::modelHPF, is_X_train::Array{Int,1}, js_X_train::Array{Int,1}, vs_X_train::Array{Float64,1})

    a, b, c, d, e, f = X.prior

    # Estimate weights among the factors
    tmpU = digamma.(X.matTheta_Shp) - log.(X.matTheta_Rte)
    tmpV = digamma.(X.matBeta_Shp) - log.(X.matBeta_Rte)
    tmpPhi = exp.(tmpU[is_X_train,:] + tmpV[js_X_train,:])
    tmpPhi = broadcast(*, tmpPhi, 1 ./ sum(tmpPhi, dims=2))


    ############################################################################
    # Update matTheta & matBeta
    # -------------------------
    tmp_inference = sum(X.matTheta[is_X_train,:] .* X.matBeta[js_X_train,:], dims=2)[:]

    for k = 1:X.K
        tensorPhi = sparse(is_X_train, js_X_train, tmpPhi[:,k] .* vs_X_train, X.M, X.N)
        X.matTheta_Shp[:, k] = a .+ sum(tensorPhi, dims=2)[:]
        X.matBeta_Shp[:, k] = d .+ sum(tensorPhi, dims=1)[:]
    end

    # Update matTheta
    X.matTheta_Rte[:,:] = broadcast(+, sum(X.matBeta, dims=1), X.vecEpsilon)
    X.matTheta[:,:] = X.matTheta_Shp ./ X.matTheta_Rte

    # Update matBeta
    X.matBeta_Rte[:,:] = broadcast(+, sum(X.matTheta, dims=1), X.vecEta)
    X.matBeta[:,:] = X.matBeta_Shp ./ X.matBeta_Rte

    # Update vecGamma & vecDelta
    X.vecEpsilon_Shp[:] .= b + X.K * a
    X.vecEpsilon_Rte[:] = c .+ sum(X.matTheta, dims=2)[:]
    X.vecEpsilon[:] = X.vecEpsilon_Shp ./ X.vecEpsilon_Rte

    X.vecEta_Shp[:] .= e + X.K * d
    X.vecEta_Rte[:] = f .+ sum(X.matBeta, dims=2)[:]
    X.vecEta[:] = X.vecEta_Shp ./ X.vecEta_Rte
    
end
