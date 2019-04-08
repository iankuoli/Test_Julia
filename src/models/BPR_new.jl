
function newBPR(K::Int, M::Int, N::Int, init_scale::Float64, usr_zeros::Array{Int,1}, itm_zeros::Array{Int,1})


    ############################################################################
    # Inference model
    # ---------------
    matTheta = ini_scale * rand(M, K) .- 0.5 * ini_scale
    matTheta[usr_zeros, :] .= 0.
    matBeta = ini_scale * rand(N, K) .- 0.5 * ini_scale
    matBeta[itm_zeros, :] .= 0.


    ############################################################################
    # Collect the model parameters into a dictionary
    # ----------------------------------------------
    return modelBPR(K, M, N, matTheta, matBeta)

end
