function model = loss_measure(model)

epoch = model.IdxEpoch;
if ~isnumeric(epoch) || ~isreal(epoch) || ~isscalar(epoch) || ~isfinite(epoch) || ...
        epoch < 1 || epoch ~= fix(epoch)
    error('NeuroFANN:InvalidParameter', 'IdxEpoch must be a positive integer.');
end
regularization = model.RegCoeff;
if ~isnumeric(regularization) || ~isreal(regularization) || ~isscalar(regularization) || ...
        ~isfinite(regularization) || regularization < 0
    error('NeuroFANN:InvalidParameter', 'RegCoeff must be finite and nonnegative.');
end
epoch = double(epoch);
regularization = double(regularization);
splits = {'Train', 'Valid'};
for index = 1:numel(splits)
    split = splits{index};
    logits = model.(['Logits', split]);
    count = model.(['Num', split]);
    targets = validatedTargets(model, split, count, logits);
    terms = max(logits, 0) - targets .* logits + log1p(exp(-abs(logits)));
    loss = sum(terms(:) / count);
    if ~isfinite(loss)
        error('NeuroFANN:NonfiniteLoss', 'The loss exceeds the finite numerical range.');
    end
    model.(['Loss', split]) = updatedHistory(model, ['Loss', split], epoch, loss);
end
if regularization == 0
    penalty = 0;
else
    scaledWeights = sqrt(regularization) * model.WeightParam;
    penalty = sum(scaledWeights .^ 2);
end
objective = model.LossTrain(epoch) + penalty;
if ~isfinite(objective)
    error('NeuroFANN:NonfiniteLoss', 'The regularized objective exceeds the finite numerical range.');
end
model.ObjectiveTrain = updatedHistory(model, 'ObjectiveTrain', epoch, objective);

end

function targets = validatedTargets(model, split, count, logits)

names = {'Yabt', 'Ymta', 'Ywmh'};
valid = isnumeric(count) && isscalar(count) && count >= 1 && ...
    (isnumeric(logits) || islogical(logits)) && isreal(logits) && ...
    isequal(size(logits), [3, count]) && all(isfinite(logits(:)));
labels = cell(3, 1);
for index = 1:3
    labels{index} = model.([names{index}, split]);
    value = labels{index};
    valid = valid && (isnumeric(value) || islogical(value)) && isreal(value) && numel(value) == count && ...
        all(isfinite(value(:))) && all(value(:) >= 0 & value(:) <= 1);
end
if ~valid
    error('NeuroFANN:InvalidTargets', ...
        'Labels and finite logits must have matching shapes, with labels in [0, 1].');
end
targets = double([reshape(labels{1}, 1, []); reshape(labels{2}, 1, []); reshape(labels{3}, 1, [])]);

end

function history = updatedHistory(model, name, epoch, value)

if isfield(model, name)
    history = double(reshape(model.(name), [], 1));
else
    history = zeros(0, 1);
end
if numel(history) < epoch
    history(numel(history) + 1:epoch, 1) = NaN;
end
history(epoch) = value;

end
