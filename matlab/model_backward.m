function model = model_backward(model)

if ~isfield(model, 'PropagationFactor')
    error('NeuroFANN:MissingForwardPass', ...
        'Run model_forward after the latest parameter reshape before model_backward.');
end
regularization = model.RegCoeff;
if ~isnumeric(regularization) || ~isreal(regularization) || ~isscalar(regularization) || ...
        ~isfinite(regularization) || regularization < 0
    error('NeuroFANN:InvalidParameter', 'RegCoeff must be finite and nonnegative.');
end
logits = model.LogitsTrain;
count = model.NumTrain;
labels = {model.YabtTrain, model.YmtaTrain, model.YwmhTrain};
valid = isnumeric(count) && isscalar(count) && count >= 1 && ...
    (isnumeric(logits) || islogical(logits)) && isreal(logits) && ...
    isequal(size(logits), [3, count]) && all(isfinite(logits(:)));
for index = 1:3
    value = labels{index};
    valid = valid && (isnumeric(value) || islogical(value)) && isreal(value) && numel(value) == count && ...
        all(isfinite(value(:))) && all(value(:) >= 0 & value(:) <= 1);
end
if ~valid
    error('NeuroFANN:InvalidTargets', ...
        'Training labels must match the logits and lie in the interval [0, 1].');
end
targets = double([reshape(labels{1}, 1, []); reshape(labels{2}, 1, []); reshape(labels{3}, 1, [])]);
residual = zeros(size(logits));
positive = logits >= 0;
negativeExp = exp(-logits(positive));
residual(positive) = (1 - targets(positive)) - negativeExp ./ (1 + negativeExp);
positiveExp = exp(logits(~positive));
residual(~positive) = positiveExp ./ (1 + positiveExp) - targets(~positive);
residual = residual / model.NumTrain;

heads = [model.Babt, model.Bmta, model.Bwmh];
headGradient = model.ZTrain * residual.';
pooledGradient = heads * residual;
attentionGradient = zeros(size(model.Aclus));
hiddenGradient = zeros(size(model.HTrain));
for cluster = 1:model.NumCluster
    members = model.ClusterMembers{cluster};
    probabilities = model.Aprob(members);
    probabilityGradient = model.HTrain(members, :) * pooledGradient(cluster, :).';
    attentionGradient(members) = probabilities .* ...
        (probabilityGradient - probabilities.' * probabilityGradient);
    hiddenGradient(members, :) = probabilities * pooledGradient(cluster, :);
end

adjoint = solver_prop(model.PropagationFactor, hiddenGradient, true);
propagationGradient = sum(adjoint .* (model.XTrain - model.HTrain), 2);
gradient = [propagationGradient; attentionGradient; headGradient(:)];
regularizationScale = sqrt(double(regularization));
model.Gradient = gradient + (2 * regularizationScale) * ...
    (regularizationScale * model.WeightParam);
if any(~isfinite(model.Gradient))
    error('NeuroFANN:NonfiniteGradient', ...
        'Backward propagation produced a nonfinite parameter gradient.');
end

end
