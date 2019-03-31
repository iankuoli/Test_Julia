include("LoadData.jl")
include("newFastHNBF.jl")

import Statistics
import SpecialFunctions

using SparseArrays

## Experimental Settings
#
# ---------- Statistics of Datasets ----------
# 1. MovieLens100K =>  M = 943     , N = 1682   , NNZ = 100K
# 2. MovieLens1M   =>  M = 6040    , N = 3900   , NNZ = 1M
# 3. LastFm2K      =>  M = 1892    , N = 17632  , NNX = 92,834
# 4. LastFm1K      =>  M = 992     , N = 174091 , NNZ = 898K
# 5. LastFm360K_2K =>  M = 2000    , N = 1682   , NNZ =
# 6. LastFm360K    =>  M = 359349  , N = 292589 , NNZ = 17,559,486
# 7. ML100KPos     =>  M = 943     , N = 1682   , NNZ = 67,331

ENV_type = Dict(1 =>"OSX", 2=>"Linux", 3=>"Windows")
DATASET = Dict(1=>"SmallToy", 2=>"SmallToyML", 3=>"ML50",
               4=>"MovieLens100K", 5=>"MovieLens1M",
               6=>"LastFm2K", 7=>"LastFm1K", 8=>"EchoNest", 9=>"LastFm360K_2K",
               10=>"LastFm360K", 11=>"ML100KPos")

TRAIN_PATH = Dict("SmallToy" => "SmallToy_train.csv",
                  "SmallToyML" => "SmallToyML_train.csv",
                  "ML50" => "ML50_train.csv",
                  "MovieLens100K" => "MovieLens100K_train_v2.csv",
                  "MovieLens1M" => "MovieLens1M_train.csv",
                  "LastFm2K" => "LastFm2K_train.csv",
                  "LastFm1K" => "LastFm1K_train.csv",
                  "EchoNest" => "EchoNest_train.csv",
                  "LastFm360K_2K" => "LastFm360K_2K_train.csv",
                  "LastFm360K" => "LastFm360K_train.csv",
                  "ML100KPos" => "ml-100k/movielens-100k-train_original.txt")

TEST_PATH = Dict("SmallToy" => "SmallToy_test.csv",
                 "SmallToyML" => "SmallToyML_test.csv",
                 "ML50" => "ML50_test.csv",
                 "MovieLens100K" => "MovieLens100K_test_v2.csv",
                 "MovieLens1M" => "MovieLens1M_test.csv",
                 "LastFm2K" => "LastFm2K_test.csv",
                 "LastFm1K" => "LastFm1K_test.csv",
                 "EchoNest" => "EchoNest_test.csv",
                 "LastFm360K_2K" => "LastFm360K_2K_test.csv",
                 "LastFm360K" => "LastFm360K_test.csv",
                 "ML100KPos" => "ml-100k/movielens-100k-test_original.txt")

VALID_PATH = Dict("SmallToy" => "SmallToy_valid.csv",
                  "SmallToyML" => "SmallToyML_valid.csv",
                  "ML50" => "ML50_valid.csv",
                  "MovieLens100K" => "MovieLens100K_valid_v2.csv",
                  "MovieLens1M" => "MovieLens1M_valid.csv",
                  "LastFm2K" => "LastFm2K_valid.csv",
                  "LastFm1K" => "LastFm1K_valid.csv",
                  "EchoNest" => "EchoNest_valid.csv",
                  "LastFm360K_2K" => "LastFm360K_2K_valid.csv",
                  "LastFm360K" => "LastFm360K_valid.csv",
                  "ML100KPos" => "ml-100k/movielens-100k-valid_original.txt")


ENTIROMENT = ENV_type[1]
DATA = DATASET[10]

NUM_RUNS = 1
likelihood_step = 10
check_step =20

ini_scale = 0.001
  data_type = 0; # 1: implicit counts ; 2: ratings; 3: binary implicit
if DATA == "SmallToy"
    data_type = 1
    MaxItr = 100
    Ks = [4]
    topK = [1, 2, 3, 5]
    G_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
               200, 1e6, 200, 1e6, 200, 1e4]
elseif DATA == "SmallToy"
    data_type = 2
    MaxItr = 400
    Ks = [4]
    topK = [1, 2, 3, 5]
    G_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
               200, 1e6, 200, 1e6, 200, 1e4]
elseif DATA == "ML50"
    data_type = 2
    MaxItr = 400
    Ks = [4]
    topK = [1, 2, 3, 5]
    G_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
               200, 1e6, 200, 1e6, 200, 1e4]
elseif DATA == "MovieLens100K"
    data_type = 2
    MaxItr = 400
    Ks = [20]
    topK = [5, 10, 15, 20]
    G_prior = [3., 1., 0.1, 3., 1., 0.1,
               1e2, 1e6, 1e2, 1e6, 2., 1e4]
elseif DATA == "MovieLens1M"
    data_type = 2
    MaxItr = 400
    Ks = [20]
    topK = [5, 10, 15, 20];
    G_prior = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3,
               1e2, 1e6, 1e2, 1e6, 2, 1e4]
elseif DATA == "LastFm2K"
    data_type = 1
    Ks = [20]
    topK = [5, 10, 15, 20]
    MaxItr = 400
    G_prior = [3, 1, 0.1, 3, 1, 0.1,
               1e2, 1e8, 1e2, 1e8, 200, 1e4]
elseif DATA == "LastFm1K"
    data_type = 1
    Ks = [20]
    topK = [5, 10, 15, 20]
    MaxItr = 400
    G_prior = [3, 1, 0.1, 3, 1, 0.1,
               1e2, 1e8, 1e2, 1e8, 200, 1e4]
elseif DATA == "EchoNest"
    data_type = 1
    MaxItr = 400
    Ks = [20]
    topK = [5, 10, 15, 20]
    G_prior = [3, 1, 0.1, 3, 1, 0.1,
               1e4, 1e8, 1e4, 1e8, 200, 1e4]
elseif DATA == "LastFm360K"
    data_type = 1
    MaxItr = 200
    Ks = [20]
    topK = [5, 10, 15, 20];
    G_prior = [3, 1, 0.1, 3, 1, 0.1,
               1e2, 1e6, 1e5, 1e9, 200, 1e4]
elseif DATA == "LastFm360K_2K"
    data_type = 1
    MaxItr = 300
    Ks = [20]
    topK = [5, 10, 15, 20]
    G_prior = [3, 1, 0.1, 3, 1, 0.1,
               1e2, 1e8, 1e2, 1e8, 200, 1e4]
elseif DATA == "ML100KPos"
    data_type = 3
    MaxItr = 1200
    Ks = [20]
    topK = [5, 10, 15, 20]
    G_prior = [3, 1, 0.1, 3, 1, 0.1,
               1e2, 1e8, 1e2, 1e8, 200, 1e4]
end

if DATA == "MovieLens100K" || DATA == "MovieLens1M" || DATA == "ML100KPos"
  matPrecNRecall = zeros(Float32, NUM_RUNS*length(Ks), length(topK)*8)
else
  matPrecNRecall = zeros(Float32, NUM_RUNS*length(Ks), length(topK)*6)
end

## Load Data
if ENV == "Linux"
  env_path = "/home/ian/Dataset/"
elseif ENV == "OSX"
  env_path = "/Users/iankuoli/Dataset/"
elseif ENV == "Windows"
  env_path = "C:/dataset/"
end

train_path = TRAIN_PATH[DATA]
valid_path = VALID_PATH[DATA]
test_path = TEST_PATH[DATA]
M, N = LoadUtilities(env_path +　train_path, env_path + test_path, env_path + valid_path)

if max(max(matX_train)) == 10
    matX_train = matX_train / 2;
    matX_test = matX_test / 2;
    matX_valid = matX_valid / 2;
end

usr_zeros = sum(matX_train, 2)==0;
itm_zeros = sum(matX_train, 1)==0;

usr_idx = 1:M
itm_idx = 1:N
deleteat!(usr_idx, sum(matX_train[usr_idx,:], dims=2).==0)
deleteat!(usr_idx, sum(matX_train[:,itm_idx], dims=1).==0)
is_X_train, js_X_train, vs_X_train = findnz(matX_train[usr_idx, itm_idx])
is_X_valid, js_X_valid, vs_X_valid = findnz(matX_valid)
is_X_test, js_X_test, vs_X_test = findnz(matX_test)


################################################################################
# Experiments
# -----------
for kk = 1:length(Ks)
    for num = 1:NUM_RUNS
        #
        # Paramter settings ---------------------------------------------------
        #
        K = Ks[kk]
        usr_batch_size = M

        valid_precision = zeros(Float32, ceil(MaxItr/check_step), length(topK))
        valid_recall = zeros(Float32, ceil(MaxItr/check_step), length(topK))
        valid_nDCG = zeros(Float32, ceil(MaxItr/check_step), length(topK))
        valid_MRR = zeros(Float32, ceil(MaxItr/check_step), length(topK))

        test_precision = zeros(Float32, ceil(MaxItr/check_step), length(topK))
        test_recall = zeros(Float32, ceil(MaxItr/check_step), length(topK))
        test_nDCG = zeros(Float32, ceil(MaxItr/check_step), length(topK))
        test_MRR = zeros(Float32, ceil(MaxItr/check_step), length(topK))

        train_poisson = zeros(Float32, ceil(MaxItr/likelihood_step), 2)
        test_poisson = zeros(Float32, ceil(MaxItr/likelihood_step), 2)
        valid_poisson = zeros(Float32, ceil(MaxItr/likelihood_step), 2)

        vecD_tmpX = zeros(Float32, ceil(MaxItr/likelihood_step), 3)

        #
        # Model initialization -------------------------------------------------
        #
        if data_type == 1
            # implicit count
            G_prior(1:6) = [30, 1*K, 0.1*sqrt(K),
                            30, 1*K, 0.1*sqrt(K)]
        elseif data_type == 2
        elseif data_type == 3
            # binary implicit
            G_prior(1:6) = [30, 1*K, 0.1*sqrt(K),
                            30, 1*K, 0.1*sqrt(K)]
        elseif data_type == 4
            # sparse implicit count
            G_prior(1:6) = [3, 1*K, 0.1*sqrt(K),
                            3, 1*K, 0.1*sqrt(K)]
        end

        dictModelParams = newFastHNBF(ini_scale, is_X_train, js_X_train, vs_X_train)

        #
        # Model Training per Epoch ---------------------------------------------
        #
        itr = 0
        IsConverge = false
        total_time = 0
        while IsConverge == false
            t = cputime
            itr = itr + 1

            @printf("Itr: %d  K = %d  ==> ", itr, K)
            @printf("subPredict_X: ( %d , %d ) , nnz = %d\n", usr_idx_len, itm_idx_len, nnz(matX_train[usr_idx, itm_idx]))

            # Update the model
            Learn_FastHNBF(dictModelParams, usr_idx, itm_idx)

            total_time = total_time + (cputime - t)

            # Calculate log likelihood of Poisson and Negative Binomial
            if likelihood_step > 0 && itr % likelihood_step == 0

                tmpX = sum(G_matTheta[is_X_train,:] .* G_matBeta[js_X_train,:], dims=2)
                tmpXX = vec_matD_ui .* tmpX
                vstrain_poisson = Evaluate_LogLikelihood_Poisson(vs_X_train, tmpX)
                vstrain_neg_binomoial = Evaluate_LogLikelihood_Poisson(vs_X_train, tmpXX)

                tmpX = sum(G_matTheta[is_X_test,:] .* G_matBeta[js_X_test,:], dims=2)
                tmpXX = sum(G_matGamma[is_X_test,:] .* G_matDelta[js_X_test,:], dims=2) .* tmpX
                vstest_poisson = Evaluate_LogLikelihood_Poisson(vs_X_test, tmpX)
                vstest_neg_binomoial = Evaluate_LogLikelihood_Poisson(vs_X_test, tmpXX)

                tmpX = sum(G_matTheta[is_X_valid,:] .* G_matBeta[js_X_valid,:], dims=2);
                tmpXX = sum(G_matGamma[is_X_valid,:] .* G_matDelta[js_X_valid,:], dims=2) .* tmpX
                vsvalid_poisson = Evaluate_LogLikelihood_Poisson(vs_X_valid, tmpX);
                vsvalid_neg_binomoial = Evaluate_LogLikelihood_Poisson(vs_X_valid, tmpXX);

                @printf("Train Loglikelihood of Poisson: %f\n", Statistics.mean(vstrain_poisson))
                @printf("Train Loglikelihood of NegBinomial: %f\n", Statistics.mean(vstrain_neg_binomoial))
                @printf("Valid Loglikelihood of Poisson: %f\n", Statistics.mean(vsvalid_poisson))
                @printf("Valid Loglikelihood of NegBinomial: %f\n", Statistics.mean(vsvalid_neg_binomoial))
                @printf(" Test Loglikelihood of Poisson: %f\n", Statistics.mean(vstest_poisson))
                @printf(" Test Loglikelihood of NegBinomial: %f\n", Statistics.mean(vstest_neg_binomoial))

                l_step_indx = itr/likelihood_step;
                train_poisson[l_step_indx, 1] = Statistics.mean(vstrain_poisson)
                train_poisson[l_step_indx, 2] = Statistics.mean(vstrain_neg_binomoial)
                test_poisson[l_step_indx, 1] = Statistics.mean(vstest_poisson)
                test_poisson[l_step_indx, 2] = Statistics.mean(vstest_neg_binomoial)
                valid_poisson[l_step_indx, 1] = Statistics.mean(vsvalid_poisson)
                valid_poisson[l_step_indx, 2] = Statistics.mean(vsvalid_neg_binomoial)
           end
        end
    end
end
