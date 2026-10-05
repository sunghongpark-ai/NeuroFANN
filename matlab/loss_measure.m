function model = loss_measure(model)

validateattributes(model.IdxEpoch, {'numeric'}, ...
    {'scalar', 'integer', 'positive', 'finite'}, mfilename, 'IdxEpoch');
validateattributes(model.RegCoeff, {'numeric'}, ...
    {'scalar', 'real', 'nonnegative', 'finite'}, mfilename, 'RegCoeff');
splits = {'Train', 'Valid'};
for index = 1:numel(splits)
    split = splits{index};
    logits = model.(['Logits', split]);
    targets = [reshape(model.(['Yabt', split]), 1, []); ...
        reshape(model.(['Ymta', split]), 1, []); ...
        reshape(model.(['Ywmh', split]), 1, [])];
    count = model.(['Num', split]);
    if count < 1 || ~isequal(size(logits), size(targets)) || ...
            size(logits, 2) ~= count || any(~isfinite(targets(:))) || ...
            any(targets(:) < 0 | targets(:) > 1) || any(~isfinite(logits(:)))
        error('NeuroFANN:InvalidTargets', ...
            'Labels and finite logits must have matching shapes, with labels in [0, 1].');
    end
    terms = max(logits, 0) - targets .* logits + log1p(exp(-abs(logits)));
    loss = sum(terms(:) / count);
    if ~isfinite(loss)
        error('NeuroFANN:NonfiniteLoss', 'The loss exceeds the finite numerical range.');
    end
    model.(['Loss', split])(model.IdxEpoch) = loss;
end
if model.RegCoeff == 0
    penalty = 0;
else
    scaledWeights = sqrt(model.RegCoeff) * model.WeightParam;
    penalty = sum(scaledWeights .^ 2);
end
objective = model.LossTrain(model.IdxEpoch) + penalty;
if ~isfinite(objective)
    error('NeuroFANN:NonfiniteLoss', 'The regularized objective exceeds the finite numerical range.');
end
model.ObjectiveTrain(model.IdxEpoch) = objective;

end
