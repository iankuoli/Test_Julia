#
# Learning phase of Generator
# The Parameters of generator are updated by (stochastic) variaitonal inference
# J = -(D(G(z)) - 1)^2 + p(\omage_d \vert \alpha_d)
# size(matSamples) = (usr_idx, itm_idx * R), where R is the number of samples per entry
#
function Learn_BPR(X::modelBPR, lr::Float64, lambda::Float64, usr_idx::Array{Int,1},
                   is_X_train::Array{Int,1}, js_X_train::Array{Int,1}, vs_X_train::Array{Float64,1})

    Itr_step = Int(floor(nnz(matX_train) / length(usr_idx)))
    i_idx = zeros(Int, length(usr_idx), Itr_step)
    j_idx = zeros(Int, length(usr_idx), Itr_step)
    for uu = 1:length(usr_idx)
        uu_nonzeros = findall(x->x>0, matX_train[usr_idx[uu], :])
        uu_zeros = findall(x->x==0, matX_train[usr_idx[uu], :])

        i_idx[uu, :] = StatsBase.sample(uu_nonzeros, Itr_step, replace=true)
        j_idx[uu, :] = StatsBase.sample(uu_zeros, Itr_step, replace=true)
    end

    for batch_step = 1:Itr_step

        x_cap_uij = sum(X.matTheta[usr_idx,:] .* (X.matBeta[i_idx[:, batch_step],:] .- X.matBeta[j_idx[:, batch_step], :]), dims=2)
        tmp_uij = 1 ./ (1 .+ exp.(x_cap_uij))
        tmp_uij[isnan.(tmp_uij)] .= 1.

        ############################################################################
        # Update X.matTheta
        # -------------------------
        X.matTheta[usr_idx,:] += lr * (broadcast(*, X.matBeta[i_idx[:, batch_step],:] .- X.matBeta[j_idx[:, batch_step],:], tmp_uij) - lambda * X.matTheta[usr_idx,:])
        X.matTheta[X.matTheta .> 100] .= 100.
        X.matTheta[X.matTheta .< -100] .= -100.

        X.matBeta[i_idx[:, batch_step],:] += lr * (broadcast(*, X.matTheta[usr_idx,:], tmp_uij) .- lambda * X.matBeta[i_idx[:, batch_step],:])
        X.matBeta[j_idx[:, batch_step],:] += lr * (broadcast(*, -X.matTheta[usr_idx,:], tmp_uij) .- lambda * X.matBeta[j_idx[:, batch_step],:])
        X.matBeta[X.matBeta .> 100] .= 100.
        X.matBeta[X.matBeta .< -100] .= -100.
    end
end
