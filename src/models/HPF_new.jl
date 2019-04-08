
function newHPF(K::Int, M::Int, N::Int, init_scale::Float64, usr_zeros::Array{Int,1}, itm_zeros::Array{Int,1},
                 matX_train::SparseMatrixCSC{Float64,Int}, prior::Array{Float64,1})

    (is_X_train, js_X_train, vs_X_train) = findnz(matX_train)

    a, b, c, d, e, f = prior


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
    # Collect the model parameters into a dictionary
    # ----------------------------------------------
    return modelHPF(K, M, N, prior,
                    matTheta, matTheta_Shp, matTheta_Rte,
                    vecEpsilon, vecEpsilon_Shp, vecEpsilon_Rte,
                    matBeta, matBeta_Shp, matBeta_Rte,
                    vecEta,  vecEta_Shp,  vecEta_Rte)

end
