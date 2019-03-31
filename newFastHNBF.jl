function newFastHNBF(init_delta::Float32, prior::Array{Float32,1}, usr_zeros::Array{Float32,1}, itm_zeros::Array{Float32,1})
    %% Intialization
    [is_X_train, js_X_train, vs_X_train] = find(matX_train)

    a, b, c, d, e, f, mu, h_mu, pi, h_pi, R, h_R = prior

    vecEpsilon_Shp = init_delta * rand(Float32, M, 1) + b
    vecEpsilon_Rte = init_delta * rand(Float32, M, 1) + c
    vecEpsilon = vecEpsilon_Shp ./ vecEpsilon_Rte

    vecEta_Shp = init_delta * rand(Float32, N, 1) + e
    vecEta_Rte = init_delta * rand(Float32, N, 1) + f
    vecEta = vecEta_Shp ./ vecEta_Rte

    matBeta_Shp = broacast(+, init_delta * rand(Float32, N, K), a)
    matBeta_Rte = broacast(+, init_delta * rand(Float32, N, K), vecEta)
    matBeta = matBeta_Shp ./ matBeta_Rte
    matBeta_Shp[itm_zeros, :] = 0
    matBeta_Rte[itm_zeros, :] = 0
    matBeta[itm_zeros, :] = 0

    matTheta_Shp = broacast(+, init_delta * rand(Float32, M, K), d)
    matTheta_Rte = broacast(+, init_delta * rand(Float32, M, K), vecEpsilon)
    matTheta = matTheta_Shp ./ matTheta_Rte
    matTheta_Shp[usr_zeros, :] = 0
    matTheta_Rte[usr_zeros, :] = 0
    matTheta[usr_zeros, :] = 0

    matGamma_Shp = broacast(+, init_delta*1000 * rand(Float32, M, K), h_mu)
    matGamma_Rte = broacast(+, init_delta*1000 * rand(Float32, M, K), 100*h_mu)
    matGamma = matGamma_Shp ./ matGamma_Rte
    matGamma_Shp[usr_zeros, :] = 0
    matGamma_Rte[usr_zeros, :] = 0
    matGamma[usr_zeros, :] = 0

    matDelta_Shp = broacast(+, init_delta*1000 * rand(Float32, N, K), h_pi)
    matDelta_Rte = broacast(+, init_delta*1000 * rand(Float32, N, K), 100*h_pi)
    matDelta = matDelta_Shp ./ matDelta_Rte
    matDelta_Shp[itm_zeros, :] = 0
    matDelta_Rte[itm_zeros, :] = 0
    matDelta[itm_zeros, :] = 0

    vec_matR_ui = 1/K * sum(matGamma[is_X_train,:] .* matDelta[js_X_train,:], 2)
    vec_matR_ui_shp = R + vec_matR_ui .* (log.(vec_matR_ui) - digamma.(vec_matR_ui))
    vec_matR_ui_rte = R ./ h_R + init_delta*1000 * rand(Float32, length(is_X_train), 1)

    vec_matD_ui_shp = vec_matR_ui + vs_X_train
    vec_matD_ui_rte = vec_matR_ui + sum(matTheta[is_X_train,:] .* matBeta[js_X_train,:], 2)
    vec_matD_ui = ones(Float32, length(vec_matR_ui), 1)

    vecMu_Shp = mu + init_delta * rand(Float32, M, 1)
    vecMu_Rte = mu / h_mu + init_delta/1e5 * rand(Float32, M, 1)
    vecMu = vecMu_Shp ./ vecMu_Rte

    vecPi_Shp = pi + init_delta * rand(Float32, N, 1)
    vecPi_Rte = pi / h_pi + init_delta/1e5 * rand(Float32, N, 1)
    vecPi = vecPi_Shp ./ vecPi_Rte

    model_params = Dict("vecEpsilon" => vecEpsilon, "vecEpsilon_Shp" => vecEpsilon_Shp, "vecEpsilon_Rte" => vecEpsilon_Rte,
                        "vecEta" => vecEta, "vecEta_Shp" => vecEta_Shp, "vecEta_Rte" => vecEta_Rte,
                        "matTheta" => matTheta, "matTheta_Shp" => matTheta_Shp, "matTheta_Rte" => matTheta_Rte,
                        "matBeta" => matBeta, "matBeta_Shp" => matBeta_Shp, "matBeta_Rte" => matBeta_Rte,
                        "vecMu" => vecMu, "vecMu_Shp" => vecMu_Shp, "vecMu_Rte" => vecMu_Rte,
                        "vecPi" => vecPi, "vecPi_Shp" => vecPi_Shp, "vecPi_Rte" => vecPi_Rte,
                        "matGamma" => matGamma, "matGamma_Shp" => matGamma_Shp, "matGamma_Rte" => matGamma_Rte,
                        "matDelta" => matDelta, "matDelta_Shp" => matDelta_Shp, "matDelta_Rte" => matDelta_Rte,
                        "vec_matD_ui" => vec_matD_ui, "vec_matD_ui_shp" => vec_matD_ui_shp, "vec_matD_ui_rte" => vec_matD_ui_rte,
                        "vec_matR_ui" => vec_matR_ui, "vec_matR_ui_shp" => vec_matR_ui_shp, "vec_matR_ui_rte" => vec_matR_ui_rte,
                        "prior" => prior)

    return model_params
end
