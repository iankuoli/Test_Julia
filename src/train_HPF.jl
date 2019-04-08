using SpecialFunctions
using SparseArrays
using Printf
using Distributed

import StatsBase
import Statistics
import DelimitedFiles

include("models/HPF_model.jl")
include("models/HPF_new.jl")
include("models/HPF_learn.jl")

include("utils/load_data.jl")
include("evals/evaluate.jl")
include("evals/eval_probs.jl")

include("experiment_setting.jl")


################################################################################
# Experimental Settings
# ---------------------
#
# ------------------- Statistics of Datasets -------------------
# 1.  SmallToy
# 2.  SmallToyML
# 3.  ML50
# 4.  MovieLens100K =>  M = 943     , N = 1682   , NNZ = 100K
# 5.  MovieLens1M   =>  M = 6040    , N = 3900   , NNZ = 1M
# 6.  LastFm2K      =>  M = 1892    , N = 17632  , NNX = 92,834
# 7.  LastFm1K      =>  M = 992     , N = 174091 , NNZ = 898K
# 8.  EchoNest      =>  M = ???     , N = ???    , NNZ = ???
# 9.  LastFm360K_2K =>  M = 2000    , N = 1682   , NNZ =
# 10. LastFm360K    =>  M = 359349  , N = 292589 , NNZ = 17,559,486
# 11. ML100KPos     =>  M = 943     , N = 1682   , NNZ = 67,331


ENVIROMENT = Env_type(3)
DATA = Dataset(6)
data_type, MaxItr, Ks, topK, prior = ExpSetting_HPF(DATA)

NUM_RUNS = 1
likelihood_step = 10
check_step =20
ini_scale = 0.001

use_ndcg = DATA == "MovieLens100K" || DATA == "MovieLens1M" || DATA == "ML100KPos"
if use_ndcg
  matPrecNRecall = zeros(NUM_RUNS*length(Ks), (length(topK)*3+1)*2)
else
  matPrecNRecall = zeros(NUM_RUNS*length(Ks), (length(topK)*2+1)*2)
end


################################################################################
# Load Data
# ---------
train_path, valid_path, test_path = Data_path(ENVIROMENT, DATA)

matX_train, matX_test, matX_valid, M, N = LoadUtilities(train_path, test_path, valid_path)
if DATA == "ML100KPos"
    matX_train[matX_train < 4.] = 0.
    matX_train[matX_train > 3.99] = 5.
    matX_test[matX_test < 4.] = 0.
    matX_test[matX_test > 3.99] = 5.
    matX_valid[matX_valid < 4.]= 0.
    matX_valid[matX_valid > 3.99] = 5.
end

# Find inactive user index and item index
usr_zeros = findall(x->x==0, sum(matX_train, dims=2)[:])
itm_zeros = findall(x->x==0, sum(matX_train, dims=1)[:])

# Find active user index and item index
usr_idx = findall(x->x>0, sum(matX_train, dims=2)[:])
itm_idx = findall(x->x>0, sum(matX_train, dims=1)[:])
usr_idx_len = length(usr_idx)
itm_idx_len = length(itm_idx)

# Find the indices of nonzero entries
is_X_train, js_X_train, vs_X_train = findnz(matX_train)
is_X_valid, js_X_valid, vs_X_valid = findnz(matX_valid)
is_X_test, js_X_test, vs_X_test = findnz(matX_test)


################################################################################
# Experiments
# -----------
for kk = 1:length(Ks)
    for num = 1:NUM_RUNS

        K = Ks[kk]

        ########################################################################
        # Initialize record containers
        # ----------------------------
        valid_precision = zeros(Int(ceil(MaxItr/check_step)), length(topK))
        valid_recall = zeros(Int(ceil(MaxItr/check_step)), length(topK))
        valid_nDCG = zeros(Int(ceil(MaxItr/check_step)), length(topK))
        valid_MRR = zeros(Int(ceil(MaxItr/check_step)), 1)

        test_precision = zeros(Int(ceil(MaxItr/check_step)), length(topK))
        test_recall = zeros(Int(ceil(MaxItr/check_step)), length(topK))
        test_nDCG = zeros(Int(ceil(MaxItr/check_step)), length(topK))
        test_MRR = zeros(Int(ceil(MaxItr/check_step)), 1)

        train_poisson = zeros(Int(ceil(MaxItr/likelihood_step)))
        test_poisson = zeros(Int(ceil(MaxItr/likelihood_step)))
        valid_poisson = zeros(Int(ceil(MaxItr/likelihood_step)))


        ########################################################################
        # Model initialization
        # --------------------
        X = newHPF(K, M, N, ini_scale, usr_zeros, itm_zeros, matX_train, prior)

        itr = 0
        last_poisson_likeli = -1e10
        IS_CONVERGE = false
        while IS_CONVERGE == false
            itr = itr + 1

            @printf("Num: %d , Itr: %d  K = %d  ==> ", num, itr, K)
            @printf("subPredict_X: ( %d , %d ) , nnz = %d \n",
                    usr_idx_len, itm_idx_len, nnz(matX_train))

            # Train model
            Learn_HPF(X, is_X_train, js_X_train, vs_X_train)


            ####################################################################
            # Calculate log likelihood of Poisson and Negative Binomial
            # ---------------------------------------------------------
            if likelihood_step > 0 && itr % likelihood_step == 0

                tmpX = sum(X.matTheta[is_X_train,:] .* X.matBeta[js_X_train,:], dims=2)[:]
                vstrain_poisson = Evaluate_LogLikelihood_Poisson(vs_X_train, tmpX)

                tmpX = sum(X.matTheta[is_X_test,:] .* X.matBeta[js_X_test,:], dims=2)[:]
                vstest_poisson = Evaluate_LogLikelihood_Poisson(vs_X_test, tmpX)

                tmpX = sum(X.matTheta[is_X_valid,:] .* X.matBeta[js_X_valid,:], dims=2)[:]
                vsvalid_poisson = Evaluate_LogLikelihood_Poisson(vs_X_valid, tmpX)

                @printf("Train Loglikelihood of Poisson: %0.4f\n", Statistics.mean(vstrain_poisson))
                @printf("Valid Loglikelihood of Poisson: %0.4f\n", Statistics.mean(vsvalid_poisson))
                @printf(" Test Loglikelihood of Poisson: %0.4f\n", Statistics.mean(vstest_poisson))

                l_step_indx = Int(itr / likelihood_step)
                train_poisson[l_step_indx] = Statistics.mean(vstrain_poisson)
                test_poisson[l_step_indx] = Statistics.mean(vstest_poisson)
                valid_poisson[l_step_indx] = Statistics.mean(vsvalid_poisson)
            end


            ####################################################################
            # Calculate precision, recall, MRR, and nDCG
            # ------------------------------------------
            if check_step > 0 && mod(itr, check_step) == 0

                # Calculate the metrics on validation set
                @printf("Validation ... \n")
                indx = Int(itr / check_step)
                if usr_idx_len > 5000 && itm_idx_len > 20000
                    user_probe = StatsBase.sample(usr_idx, min(usr_idx_len, 5000), replace=false)
                else
                    user_probe = usr_idx
                end

                dictResult = Evaluate(matX_valid[user_probe,:], matX_train[user_probe,:],
                                      X.matTheta[user_probe,:], X.matBeta, topK, ndcg=use_ndcg)

                valid_precision[indx,:] = dictResult["Precision"]
                valid_recall[indx,:] = dictResult["Recall"]
                valid_MRR[indx] = dictResult["MRR"]
                @printf("validation precision: %0.4f, %0.4f, %0.4f, %0.4f\n",
                        valid_precision[indx, 1], valid_precision[indx, 2], valid_precision[indx, 3], valid_precision[indx, 4])
                @printf("validation recall: %0.4f, %0.4f, %0.4f, %0.4f\n",
                        valid_recall[indx, 1], valid_recall[indx, 2], valid_recall[indx, 3], valid_recall[indx, 4])
                if use_ndcg
                    valid_nDCG[indx,:] = dictResult["nDCG"]
                    @printf("validation nDCG: %0.4f, %0.4f, %0.4f, %0.4f\n",
                                        valid_nDCG[indx, 1], valid_nDCG[indx, 2], valid_nDCG[indx, 3], valid_nDCG[indx, 4])
                end

                # Calculate the metrics on testing set
                @printf("Testing ... \n");
                indx = Int(itr / check_step)
                if usr_idx_len > 5000 && itm_idx_len > 20000
                    user_probe = StatsBase.sample(usr_idx, min(usr_idx_len, 5000), replace=false)
                else
                    user_probe = usr_idx
                end
                dictResult = Evaluate(matX_test[user_probe,:]+matX_valid[user_probe,:], matX_train[user_probe,:],
                                      X.matTheta[user_probe,:], X.matBeta, topK, ndcg=use_ndcg)

                test_precision[indx,:] = dictResult["Precision"]
                test_recall[indx,:] = dictResult["Recall"]
                test_MRR[indx] = dictResult["MRR"]
                @printf("testing precision: %0.4f, %0.4f, %0.4f, %0.4f\n",
                        test_precision[indx, 1], test_precision[indx, 2], test_precision[indx, 3], test_precision[indx, 4])
                @printf("testing recall: %0.4f, %0.4f, %0.4f, %0.4f\n",
                        test_recall[indx, 1], test_recall[indx, 2], test_recall[indx, 3], test_recall[indx, 4])
                if use_ndcg
                    test_nDCG[indx,:] = dictResult["nDCG"]
                    @printf("testing nDCG: %0.4f, %0.4f, %0.4f, %0.4f\n",
                            test_nDCG[indx, 1], test_nDCG[indx, 2], test_nDCG[indx, 3], test_nDCG[indx, 4])
                end
            end

            if itr >= MaxItr
                IS_CONVERGE = true
            end
            if itr % likelihood_step == 0
                if itr > 180 && Statistics.mean(vsvalid_poisson) - last_poisson_likeli < 0.2
                    IS_CONVERGE = true
                end
                last_poisson_likeli = Statistics.mean(vsvalid_poisson);
            end
        end


        ########################################################################
        # Record the experimental result
        # ------------------------------
        Best_matTheta = X.matTheta
        Best_matBeta = X.matBeta

        dictResult = Evaluate(matX_test[usr_idx,:]+matX_valid[usr_idx,:], matX_train[usr_idx,:],
                              X.matTheta[usr_idx,:], X.matBeta, topK, ndcg=use_ndcg)

        total_test_precision = dictResult["Precision"]
        total_test_recall = dictResult["Recall"]
        total_test_MRR = dictResult["MRR"]
        @printf("total testing precision: %0.4f, %0.4f, %0.4f, %0.4f\n",
                total_test_precision[1], total_test_precision[2], total_test_precision[3], total_test_precision[4])
        @printf("total testing recall: %0.4f, %0.4f, %0.4f, %0.4f\n",
                total_test_recall[1], total_test_recall[2], total_test_recall[3], total_test_recall[4])

        if use_ndcg
            total_test_nDCG = dictResult["nDCG"]
            @printf("total testing nDCG: %0.4f, %0.4f, %0.4f, %0.4f\n",
                    total_test_nDCG[1], total_test_nDCG[2], total_test_nDCG[3], total_test_nDCG[4])
        end

        if use_ndcg
            matPrecNRecall[(kk-1)*NUM_RUNS+num,1:(length(topK)*3+1)] = vcat(total_test_precision, total_test_recall, total_test_nDCG, total_test_MRR)
            matPrecNRecall[(kk-1)*NUM_RUNS+num,(length(topK)*3+2):end] = vcat(valid_precision[end,:], valid_recall[end,:], valid_nDCG[end,:], valid_MRR[end,:])
        else
            matPrecNRecall[(kk-1)*NUM_RUNS+num,1:(length(topK)*2+1)] = vcat(total_test_precision, total_test_recall, total_test_MRR)
            matPrecNRecall[(kk-1)*NUM_RUNS+num,(length(topK)*2+2):end] = vcat(valid_precision[end,:], valid_recall[end,:], valid_MRR[end])
        end
    end
end
