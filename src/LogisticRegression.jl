module LogisticRegression
using LinearAlgebra
using Distributions
using Random
using UnicodePlots
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
function logistic_regression(feats, labels; num_steps=20000, learning_rate=1e-2, add_intercept=false)
    if add_intercept == true
        intercept = ones(size(feats)[1])
        feats = [intercept feats]
    end
    # should have a weight for each column
    weights = zeros(size(feats)[2]) 
    
    for step=1:num_steps
        scores = feats * weights
        predictions = par_sigmoid(scores)
        
        # update using the gradient
        error_signal = labels - predictions
        gradient = transpose(feats) * error_signal
        weights += learning_rate * gradient
        if step % 10000 == 0
            println(log_likelihood(feats, labels, weights))
        end
    end
    return weights
end
function test_log_reg()
    num_instances = 5000

    x1_v1 = rand(Normal(0, 2), num_instances)
    x1_v2 = rand(Normal(0, .3), num_instances)
    x2_v1 = rand(Normal(4, 1), num_instances)
    x2_v2 = rand(Normal(2, .3), num_instances)
    x1 = [x1_v1 x1_v2]
    x2 = [x2_v1 x2_v2]
    # display the dataset we just made
    plt = scatterplot(x1[:, 1], x2[:, 2], ylim=[-2,3])
    scatterplot!(plt, x2[:, 1], x2[:, 2], color=:blue)
    # combine these two separate
    # normal distributions
    data = [x1 ; x2]
    data_with_intercept = [ones(size(data)[1]) data]
    labels = [zeros(num_instances) ; ones(num_instances)]
    weights = logistic_regression(data, labels, num_steps=300000, learning_rate=5e-5, add_intercept=true)
    final_scores = data_with_intercept * weights
    preds = round.(par_sigmoid(final_scores))
    println("Accuracy: ")
    acc = sum(preds .== labels)/length(labels)
    println(acc)
    return acc
end
end # module
