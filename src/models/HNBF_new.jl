
function newHNBF(K::Int, M::Int, N::Int, init_scale::Float64, usr_zeros::Array{Int,1}, itm_zeros::Array{Int,1},
                 matX_train::SparseMatrixCSC{Float64,Int}, prior::Array{Float64,1})

    (is_X_train, js_X_train, vs_X_train) = findnz(matX_train)

    a, b, c, d, e, f, mu, h_mu, pi, h_pi, R, h_R = prior


    ############################################################################
    # Inference model
    # ---------------
    vecEpsilon_Shp = init_scale * rand(M) .+ b
    vecEpsilon_Rte = init_scale * rand(M) .+ c
    vecEpsilon = vecEpsilon_Shp ./ vecEpsilon_Rte

    vecEta_Shp = init_scale * rand(N) .+ e
    vecEta_Rte = init_scale * rand(N) .+ f
    vecEta = vecEta_Shp ./ vecEta_Rte

    matTheta_Shp = init_scale * rand(M, K) .+ a
    matTheta_Rte = broadcast(+, init_scale * rand(M, K), vecEpsilon)
    matTheta = matTheta_Shp ./ matTheta_Rte
    matTheta_Shp[usr_zeros, :] .= 0.
    matTheta_Rte[usr_zeros, :] .= 0.
    matTheta[usr_zeros, :] .= 0.

    matBeta_Shp = init_scale * rand(N, K) .+ d
    matBeta_Rte = broadcast(+, init_scale * rand(N, K), vecEta)
    matBeta = matBeta_Shp ./ matBeta_Rte
    matBeta_Shp[itm_zeros, :] .= 0.
    matBeta_Rte[itm_zeros, :] .= 0.
    matBeta[itm_zeros, :] .= 0.


    ############################################################################
    # Dispersion model for zero entries
    # ---------------------------------
    vecMu_Shp = mu .+ init_scale * rand(M)
    vecMu_Rte = mu / h_mu .+ init_scale/1e5 * rand(M)
    vecMu = vecMu_Shp ./ vecMu_Rte

    vecPi_Shp = pi .+ init_scale * rand(N)
    vecPi_Rte = pi / h_pi .+ init_scale/1e5 * rand(N)
    vecPi = vecPi_Shp ./ vecPi_Rte

    matGamma_Shp = init_scale*1000 * rand(M, K) .+ h_mu
    matGamma_Rte = init_scale*1000 * rand(M, K) .+ 100*h_mu
    matGamma = matGamma_Shp ./ matGamma_Rte
    matGamma_Shp[usr_zeros, :] .= 0
    matGamma_Rte[usr_zeros, :] .= 0
    matGamma[usr_zeros, :] .= 0

    matDelta_Shp = init_scale*1000 * rand(N, K) .+ h_pi
    matDelta_Rte = init_scale*1000 * rand(N, K) .+ 100*h_pi
    matDelta = matDelta_Shp ./ matDelta_Rte
    matDelta_Shp[itm_zeros, :] .= 0
    matDelta_Rte[itm_zeros, :] .= 0
    matDelta[itm_zeros, :] .= 0


    ############################################################################
    # Dispersion model for nonzero entries
    # ------------------------------------
    vec_matR_ui = 1/K * sum(matGamma[is_X_train,:] .* matDelta[js_X_train,:], dims=2)[:]
    vec_matR_ui_Shp = R .+ vec_matR_ui .* (log.(vec_matR_ui) - digamma.(vec_matR_ui))
    vec_matR_ui_Rte = (R / h_R) .+ (init_scale * 1000) * rand(length(is_X_train))

    vec_matD_ui_Shp = vec_matR_ui + vs_X_train
    vec_matD_ui_Rte = vec_matR_ui + sum(matTheta[is_X_train,:] .* matBeta[js_X_train,:], dims=2)[:]
    vec_matD_ui = ones(length(vec_matR_ui))


    ############################################################################
    # Collect the model parameters into a dictionary
    # ----------------------------------------------
    return modelHNBF(K, M, N, prior,
                     matTheta, matTheta_Shp, matTheta_Rte,
                     vecEpsilon, vecEpsilon_Shp, vecEpsilon_Rte,
                     matBeta, matBeta_Shp, matBeta_Rte,
                     vecEta,  vecEta_Shp,  vecEta_Rte,
                     matGamma,  matGamma_Shp,  matGamma_Rte,
                     vecMu,  vecMu_Shp,  vecMu_Rte,
                     matDelta,  matDelta_Shp,  matDelta_Rte,
                     vecPi,  vecPi_Shp,  vecPi_Rte,
                     vec_matR_ui,  vec_matR_ui_Shp,  vec_matR_ui_Rte,
                     vec_matD_ui,  vec_matD_ui_Shp,  vec_matD_ui_Rte)
    
end
