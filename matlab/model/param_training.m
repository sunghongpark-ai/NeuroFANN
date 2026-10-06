function model = param_training(model)

epochs = model.MaxEpoch;
if ~isnumeric(epochs) || ~isreal(epochs) || ~isscalar(epochs) || ~isfinite(epochs) || ...
        epochs < 1 || epochs ~= fix(epochs)
    error('NeuroFANN:InvalidParameter', 'MaxEpoch must be a positive integer scalar.');
end
epochs = double(epochs);
model.MaxEpoch = epochs;
weights = model.WeightParam;
if ~isfloat(weights) || ~isreal(weights) || ~iscolumn(weights) || isempty(weights) || any(~isfinite(weights))
    error('NeuroFANN:InvalidWeights', 'WeightParam must be a nonempty finite real floating-point column vector.');
end
if ~isfield(model, 'AdamParam')
    error('NeuroFANN:InvalidOptimizerState', 'Initialize AdamParam with init_adamopt before training.');
end
checkOptimizer(model.AdamParam, size(weights));
model.WeightParam = double(weights);
model.AdamParam.m = double(model.AdamParam.m);
model.AdamParam.v = double(model.AdamParam.v);
if ~isfield(model, 'StoreHistory') || isempty(model.StoreHistory)
    model.StoreHistory = false;
end
flag = model.StoreHistory;
if ~(isnumeric(flag) || islogical(flag)) || ~isreal(flag) || ~isscalar(flag) || ~(flag == 0 || flag == 1)
    error('NeuroFANN:InvalidParameter', 'StoreHistory must be a scalar logical flag.');
end
model.StoreHistory = logical(flag);
if model.StoreHistory
    model.WeightEpoch = cell(epochs, 1);
else
    model.WeightEpoch = cell(0, 1);
end
model.LossTrain = nan(epochs, 1);
model.LossValid = nan(epochs, 1);
model.ObjectiveTrain = nan(epochs, 1);
model.BestEpoch = 0;
model.BestValidationLoss = Inf;
bestWeights = model.WeightParam;
bestAdam = model.AdamParam;
timer = tic;

for epoch = 1:epochs
    model.IdxEpoch = epoch;
    if model.StoreHistory
        model.WeightEpoch{epoch} = model.WeightParam;
    end
    model = param_reshape_(model);
    model = model_forward(model, false);
    model = loss_measure(model);
    if ~isfinite(model.LossTrain(epoch)) || ~isfinite(model.LossValid(epoch)) || ~isfinite(model.ObjectiveTrain(epoch))
        error('NeuroFANN:NonfiniteTrainingLoss', ...
            'Nonfinite loss at epoch %d. Check input scaling, regularization, and the learning rate.', epoch);
    end
    if model.LossValid(epoch) < model.BestValidationLoss
        model.BestEpoch = epoch;
        model.BestValidationLoss = model.LossValid(epoch);
        bestWeights = model.WeightParam;
        bestAdam = model.AdamParam;
    end
    if epoch < epochs
        model = model_backward(model);
        model = param_update(model);
    end
end

model.WeightParam = bestWeights;
model.AdamParam = bestAdam;
model = param_reshape_(model);
model = model_forward(model, true);
model = model_backward(model);
model.TrainingTime = toc(timer);

end

function checkOptimizer(param, weightSize)

required = {'alpha', 'beta1', 'beta2', 'epsilon', 't', 'm', 'v'};
if ~isstruct(param) || ~isscalar(param) || ~all(isfield(param, required))
    error('NeuroFANN:InvalidOptimizerState', 'AdamParam must contain all fields created by init_adamopt.');
end
scalars = {param.alpha, param.beta1, param.beta2, param.epsilon, param.t};
for index = 1:numel(scalars)
    value = scalars{index};
    if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ~isfinite(value)
        error('NeuroFANN:InvalidOptimizerState', 'Adam hyperparameters and step must be finite real numeric scalars.');
    end
end
if param.alpha <= 0 || param.epsilon <= 0 || param.beta1 < 0 || param.beta1 >= 1 || ...
        param.beta2 < 0 || param.beta2 >= 1 || param.t < 0 || param.t >= flintmax || param.t ~= fix(param.t)
    error('NeuroFANN:InvalidOptimizerState', 'Adam hyperparameters or step are outside their valid ranges.');
end
moments = {param.m, param.v};
for index = 1:numel(moments)
    value = moments{index};
    if ~isfloat(value) || ~isreal(value) || ~isequal(size(value), weightSize) || any(~isfinite(value(:)))
        error('NeuroFANN:InvalidOptimizerState', 'Adam moments must be finite real vectors matching WeightParam.');
    end
end
if any(param.v(:) < 0)
    error('NeuroFANN:InvalidOptimizerState', 'Adam second moments must be nonnegative.');
end

end
