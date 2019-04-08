#
# Learning phase of Generator
# The Parameters of generator are updated by (stochastic) variaitonal inference
# J = -(D(G(z)) - 1)^2 + p(\omage_d \vert \alpha_d)
# size(matSamples) = (usr_idx, itm_idx * R), where R is the number of samples per entry
#
function Learn_FastHNBF(X::modelHNBF,
                        is_X_train::Array{Int,1}, js_X_train::Array{Int,1}, vs_X_train::Array{Float64,1})

    a, b, c, d, e, f = X.prior[1:6]
    mu, h_mu, pi, h_pi, R, h_R = X.prior[7:12]

    # Estimate weights among the factors
    tmpU = digamma.(X.matTheta_Shp) - log.(X.matTheta_Rte)
    tmpV = digamma.(X.matBeta_Shp) - log.(X.matBeta_Rte)
    tmpPhi = exp.(tmpU[is_X_train,:] + tmpV[js_X_train,:])
    tmpPhi = broadcast(*, tmpPhi, 1 ./ sum(tmpPhi, dims=2))


    ############################################################################
    # Update matTheta & matBeta
    # -------------------------
    tmp_inference = sum(X.matTheta[is_X_train,:] .* X.matBeta[js_X_train,:], dims=2)[:]

    # Update dispersion
    X.vec_matD_ui_Shp[:] = X.vec_matR_ui + vs_X_train
    X.vec_matD_ui_Rte[:] = X.vec_matR_ui + tmp_inference
    X.vec_matD_ui[:] = X.vec_matD_ui_Shp ./ X.vec_matD_ui_Rte

    # Update exposure count
    X.vec_matR_ui_Shp[:] = R .+ X.vec_matR_ui .* (log.(X.vec_matR_ui) - digamma.(X.vec_matR_ui))
    X.vec_matR_ui_Rte[:] = (R / h_R) .+ X.vec_matD_ui .- log.(X.vec_matD_ui) .- 1
    X.vec_matR_ui[:] = X.vec_matR_ui_Shp ./ X.vec_matR_ui_Rte

    for k = 1:X.K
        tensorPhi = sparse(is_X_train, js_X_train, tmpPhi[:,k] .* vs_X_train, X.M, X.N)
        X.matTheta_Shp[:, k] = a .+ sum(tensorPhi, dims=2)[:]
        X.matBeta_Shp[:, k] = d .+ sum(tensorPhi, dims=1)[:]

        X.matGamma_Shp[:, k] = X.vecMu
        X.matDelta_Shp[:, k] = X.vecPi
    end

    tmpD = sparse(is_X_train, js_X_train, X.vec_matD_ui - 1/X.K * sum(X.matGamma[is_X_train,:] .* X.matDelta[js_X_train,:], dims=2)[:], X.M, X.N)

    # Update matTheta
    X.matTheta_Rte[:,:] = broadcast(+, 1/X.K * (X.matGamma * (X.matDelta' * X.matBeta)) + tmpD * X.matBeta, X.vecEpsilon)
    X.matTheta[:,:] = X.matTheta_Shp ./ X.matTheta_Rte

    # Update matBeta
    X.matBeta_Rte[:,:] = broadcast(+, 1/X.K * (X.matDelta * (X.matGamma' * X.matTheta)) + tmpD' * X.matTheta, X.vecEta)
    X.matBeta[:,:] = X.matBeta_Shp ./ X.matBeta_Rte

    # Update vecGamma & vecDelta
    X.vecEpsilon_Shp[:] .= b + X.K * a
    X.vecEpsilon_Rte[:] = c .+ sum(X.matTheta, dims=2)[:]
    X.vecEpsilon[:] = X.vecEpsilon_Shp ./ X.vecEpsilon_Rte

    X.vecEta_Shp[:] .= e + X.K * d
    X.vecEta_Rte[:] = f .+ sum(X.matBeta, dims=2)[:]
    X.vecEta[:] = X.vecEta_Shp ./ X.vecEta_Rte


    ############################################################################
    # Updating Latent Factors for Dispersion
    # --------------------------------------
    tmpD = sparse(is_X_train, js_X_train, tmp_inference, X.M, X.N)

    X.matGamma_Rte[:,:] = broadcast(+, 1/X.K * (X.matTheta * (X.matBeta' * X.matDelta) - tmpD * X.matDelta), X.vecMu)
    X.matGamma[:,:] = X.matGamma_Shp ./ X.matGamma_Rte

    X.vecMu_Shp[:] = mu .+ X.K * X.vecMu .* (log.(X.vecMu) - digamma.(X.vecMu))
    X.vecMu_Rte[:] = (mu / h_mu) .+ sum(X.matGamma - log.(X.matGamma), dims=2)[:] .- X.K
    X.vecMu[:] = X.vecMu_Shp ./ X.vecMu_Rte

    X.matDelta_Rte[:,:] = broadcast(+, 1/X.K * (X.matBeta * (X.matTheta' * X.matGamma) - tmpD' * X.matGamma), X.vecPi)
    X.matDelta[:,:] = X.matDelta_Shp ./ X.matDelta_Rte

    X.vecPi_Shp[:] = pi .+ X.K * X.vecPi .* (log.(X.vecPi) - digamma.(X.vecPi))
    X.vecPi_Rte[:] = (pi / h_pi) .+ sum(X.matDelta - log.(X.matDelta), dims=2)[:] .- X.K
    X.vecPi[:] = X.vecPi_Shp ./ X.vecPi_Rte

end
