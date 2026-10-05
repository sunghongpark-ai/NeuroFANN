function model = param_training(model)

validateattributes(model.MaxEpoch, {'numeric'}, {'scalar', 'real', 'finite', 'integer', 'positive'}, mfilename, 'MaxEpoch');
if ~isfield(model, 'AdamParam') || ~isstruct(model.AdamParam) || ~isscalar(model.AdamParam)
    error('NeuroFANN:InvalidOptimizerState', 'Initialize AdamParam with init_adamopt before training.');
end
required = {'alpha', 'beta1', 'beta2', 'epsilon', 't', 'm', 'v'};
if ~all(isfield(model.AdamParam, required))
    error('NeuroFANN:InvalidOptimizerState', 'AdamParam is missing required optimizer fields. Initialize it with init_adamopt.');
end
validateattributes(model.WeightParam, {'double', 'single'}, {'column', 'nonempty', 'real', 'finite'}, mfilename, 'WeightParam');
validateattributes(model.AdamParam.alpha, {'numeric'}, {'scalar', 'real', 'finite', 'positive'}, mfilename, 'AdamParam.alpha');
validateattributes(model.AdamParam.beta1, {'numeric'}, {'scalar', 'real', 'finite', '>=', 0, '<', 1}, mfilename, 'AdamParam.beta1');
validateattributes(model.AdamParam.beta2, {'numeric'}, {'scalar', 'real', 'finite', '>=', 0, '<', 1}, mfilename, 'AdamParam.beta2');
validateattributes(model.AdamParam.epsilon, {'numeric'}, {'scalar', 'real', 'finite', 'positive'}, mfilename, 'AdamParam.epsilon');
validateattributes(model.AdamParam.t, {'numeric'}, {'scalar', 'real', 'finite', 'integer', 'nonnegative', '<', flintmax}, mfilename, 'AdamParam.t');
validateattributes(model.AdamParam.m, {'double', 'single'}, {'size', size(model.WeightParam), 'real', 'finite'}, mfilename, 'AdamParam.m');
validateattributes(model.AdamParam.v, {'double', 'single'}, {'size', size(model.WeightParam), 'real', 'finite', 'nonnegative'}, mfilename, 'AdamParam.v');
if ~isfield(model, 'StoreHistory') || isempty(model.StoreHistory)
    model.StoreHistory = false;
end
validateattributes(model.StoreHistory, {'numeric', 'logical'}, {'scalar', 'real', 'finite', 'binary'}, mfilename, 'StoreHistory');
model.StoreHistory = logical(model.StoreHistory);
if model.StoreHistory
    model.WeightEpoch = cell(model.MaxEpoch, 1);
else
    model.WeightEpoch = cell(0, 1);
end
model.LossTrain = nan(model.MaxEpoch, 1);
model.LossValid = nan(model.MaxEpoch, 1);
model.ObjectiveTrain = nan(model.MaxEpoch, 1);
model.BestEpoch = 0;
model.BestValidationLoss = Inf;
bestWeights = model.WeightParam;
bestAdam = model.AdamParam;
timer = tic;

for epoch = 1:model.MaxEpoch
    model.IdxEpoch = epoch;
    if model.StoreHistory
        model.WeightEpoch{epoch} = model.WeightParam;
    end
    model = param_reshape_(model);
    model = model_forward(model, false);
    model = loss_measure(model);
    if ~isfinite(model.LossTrain(epoch)) || ~isfinite(model.LossValid(epoch)) || ~isfinite(model.ObjectiveTrain(epoch))
        error('NeuroFANN:NonfiniteTrainingLoss', 'Nonfinite loss at epoch %d. Check input scaling, regularization, and the learning rate.', epoch);
    end
    if model.LossValid(epoch) < model.BestValidationLoss
        model.BestEpoch = epoch;
        model.BestValidationLoss = model.LossValid(epoch);
        bestWeights = model.WeightParam;
        bestAdam = model.AdamParam;
    end
    if epoch < model.MaxEpoch
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
