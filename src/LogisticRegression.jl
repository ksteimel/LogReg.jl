module LogisticRegression
using LinearAlgebra
using Distributions
using Random
# using UnicodePlots
using SparseArrays
BLAS.set_num_threads(2)

"""
This structure implements the Sklearn.jl api for 
the Logistic Regression algorithm
"""
mutable struct LogisticRegressor
    num_steps::Int64
    cutoff::Float64
    learning_rate::Float64
    add_intercept::Bool
    weights::Array{Float64, 1}
end
LogisticRegressor(; 
    num_steps=300000,
    cutoff=1e-4,
    learning_rate=5e-5,
    add_intercept=false,
    weights=Float64[]
    ) = LogisticRegressor(  num_steps,
                            cutoff,
                            learning_rate,
                            add_intercept,
                            weights )

function fit!(self::LogisticRegressor, data, labels::Array{T, 1}) where T <: Number
    weights = logistic_regression(  data, 
                                    labels, 
                                    num_steps = self.num_steps,
                                    cutoff = self.cutoff,
                                    learning_rate = self.learning_rate,
                                    add_intercept = self.add_intercept)
    self.weights = weights
end
function predict(self::LogisticRegressor, data)
    if self.add_intercept
        data = [ones(size(data)[1]) data]
    end
    final_scores = data * self.weights
    preds = round.(par_sigmoid(final_scores))
    return preds
end
function sigmoid(score::T) where T <: Number
    return 1/(1 + exp(-score))
end
function par_sigmoid(scores::Array{T}) where T <: Number
    res = zeros(length(scores))
    Threads.@threads for i in eachindex(scores)
        @inbounds res[i] = sigmoid(scores[i])
    end
    return res
end
function log_likelihood(feats, label, weights)
    scores = feats * weights
    log_likely = sum(label .* scores - log.(1 .+ exp.(scores)))
    return log_likely
end
function logistic_regression(feats, labels; num_steps=20000, cutoff=1e-1000, learning_rate=1e-2, add_intercept=false)
    if add_intercept == true
        intercept = ones(size(feats)[1])
        feats = [intercept feats]
    end
    # should have a weight for each column
    weights = zeros(size(feats)[2]) 
    scores = zeros(size(feats)[2])
    error_signal = zeros(size(feats)[2])
    last_log_likelihood_delta = 100.0
    current_log_likelihood_delta = 100.0
    last_log_likelihood = 10000.0
    current_log_likelihood = 10000.0
    for step=1:num_steps
        scores = feats * weights
        predictions = par_sigmoid(scores)
        
        # update using the gradient
        error_signal = labels - predictions
        gradient = transpose(feats) * error_signal
        weights += learning_rate * gradient
        if step % 10000 == 0
            current_log_likelihood = log_likelihood( feats, 
                                                     labels, 
                                                     weights)
            current_log_likelihood_delta = abs(current_log_likelihood - last_log_likelihood)
            println("Change in log likelihood: " * string(current_log_likelihood_delta))
            if current_log_likelihood_delta < cutoff && last_log_likelihood_delta < cutoff
                break # stop training if we haven't exceeded cutoff in 20000 iterations
            end
            last_log_likelihood_delta = current_log_likelihood_delta
            last_log_likelihood = current_log_likelihood
        end
    end
    return weights
end
function test_log_reg()
    num_instances = 5000
    # generate dense training data
    #x1_v1 = rand(Normal(0, 2), num_instances)
    #x1_v2 = rand(Normal(0, .3), num_instances)
    #x2_v1 = rand(Normal(4, 1), num_instances)
    #x2_v2 = rand(Normal(2, .3), num_instances)
    
    # generate sparse training data
    x1_v1 = sparse(rand(Binomial(4, .2), num_instances))
    x1_v2 = sparse(rand(Binomial(8, .1),
    num_instances))
    x2_v1 = sparse(rand(Binomial(4, .8), num_instances))
    x2_v2 = sparse(rand(Binomial(8, .7), num_instances))
    
    x1 = [x1_v1 x1_v2]
    x2 = [x2_v1 x2_v2]
    # generate sparse test data
    x3_v1 = sparse(rand(Binomial(4, .2), num_instances))
    x3_v2 = sparse(rand(Binomial(8, .1),
    num_instances))
    x4_v1 = sparse(rand(Binomial(4, .8), num_instances))
    x4_v2 = sparse(rand(Binomial(8, .7), num_instances))
    x3 = [x3_v1 x3_v2]
    x4 = [x4_v1 x4_v2]
    test_data = [x3 ; x4]
    test_labels = [zeros(num_instances) ; ones(num_instances)]
    # display the dataset we just made
    # plt = scatterplot(x1[:, 1], x2[:, 2], ylim=[-2,3])
    # scatterplot!(plt, x2[:, 1], x2[:, 2], color=:blue)
    # combine these two separate
    # normal distributions
    data = [x1 ; x2]
    # data_with_intercept = [ones(size(data)[1]) data]
    labels = [zeros(num_instances) ; ones(num_instances)]
    # weights = logistic_regression(data, labels, num_steps=300000, learning_rate=5e-5, add_intercept=true)
    # final_scores = data_with_intercept * weights
    # preds = round.(par_sigmoid(final_scores))
    model = LogisticRegressor(add_intercept=true)
    fit!(model, data, labels)
    test_preds = predict(model, test_data)
    # println("Accuracy on training data: ")
    # acc = sum(preds .== labels)/length(labels)
    # println(acc)
    # test_data_with_intercept = [ones(size(test_data)[1]) test_data]
    # test_scores = test_data_with_intercept * weights
    # test_preds = round.(par_sigmoid(test_scores))
    println("Accuracy on test data: ")
    acc = sum(test_preds .== test_labels)/length(test_labels)
    println(acc)
    return acc
end
export LogisticRegressor, fit!, predict
end # module
