function model = model_backward(model)

if ~isfield(model, 'PropagationFactor')
    error('NeuroFANN:MissingForwardPass', ...
        'Run model_forward after the latest parameter reshape before model_backward.');
end
targets = [model.YabtTrain(:).'; model.YmtaTrain(:).'; model.YwmhTrain(:).'];
logits = model.LogitsTrain;
if model.NumTrain < 1 || ~isequal(size(targets), size(logits)) || ...
        size(logits, 2) ~= model.NumTrain || any(~isfinite(targets(:))) || ...
        any(targets(:) < 0 | targets(:) > 1)
    error('NeuroFANN:InvalidTargets', ...
        'Training labels must match the logits and lie in the interval [0, 1].');
end
residual = zeros(size(logits), 'like', logits);
positive = logits >= 0;
negativeExp = exp(-logits(positive));
residual(positive) = (1 - targets(positive)) - negativeExp ./ (1 + negativeExp);
positiveExp = exp(logits(~positive));
residual(~positive) = positiveExp ./ (1 + positiveExp) - targets(~positive);
residual = residual / model.NumTrain;

heads = [model.Babt, model.Bmta, model.Bwmh];
headGradient = model.ZTrain * residual.';
pooledGradient = heads * residual;
attentionGradient = zeros(size(model.Aclus), 'like', model.Aclus);
hiddenGradient = zeros(size(model.HTrain), 'like', model.HTrain);
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
regularizationScale = sqrt(model.RegCoeff);
model.Gradient = gradient + (2 * regularizationScale) * ...
    (regularizationScale * model.WeightParam);
if any(~isfinite(model.Gradient))
    error('NeuroFANN:NonfiniteGradient', ...
        'Backward propagation produced a nonfinite parameter gradient.');
end

end
