
function newExpoMF(K::Int, M::Int, N::Int, ini_scale::Float64, usr_zeros::Array{Int,1}, itm_zeros::Array{Int,1})


    ############################################################################
    # Inference model
    # ---------------
    matTheta = ini_scale * rand(M, K) .- 0.5 * ini_scale
    matTheta[usr_zeros, :] .= 0.
    matBeta = ini_scale * rand(N, K) .- 0.5 * ini_scale
    matBeta[itm_zeros, :] .= 0.
    vecMu = ini_scale * rand(N)
    vecMu[itm_zeros] .= 0.

    valLambda_y = 0.1
    valLambda_theta = 0.01
    valLambda_beta = 0.01
    val_alpha1 = 1.
    val_alpha2 = 1.

    ############################################################################
    # Collect the model parameters into a dictionary
    # ----------------------------------------------
    return modelExpoMF(K, M, N, matTheta, matBeta, vecMu,
                       valLambda_y, valLambda_theta, valLambda_beta,
                       val_alpha1, val_alpha2)

end
