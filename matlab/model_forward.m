function model = model_forward(model, includeTest)

if nargin < 2
    includeTest = true;
end
validateattributes(includeTest, {'logical', 'numeric'}, ...
    {'scalar', 'real', 'finite', 'binary'}, mfilename, 'includeTest');

operator = model.Udiag + model.Lppi;
if ~isreal(operator) || any(~isfinite(nonzeros(operator)))
    error('NeuroFANN:NonfinitePropagation', ...
        'The propagation matrix must contain finite real values.');
end
factor = decomposition(operator, 'lu', 'CheckCondition', false);
reciprocalCondition = rcond(factor);
if ~isfinite(reciprocalCondition) || reciprocalCondition <= eps(class(operator))
    error('NeuroFANN:SingularPropagation', ...
        'diag(Uprot) + Lppi is singular or numerically singular (reciprocal condition %.3g).', ...
        reciprocalCondition);
end
model.PropagationFactor = factor;
model.PropagationReciprocalCondition = reciprocalCondition;

if ~isfield(model, 'ClusterAssignment') || ...
        ~isequal(size(model.ClusterAssignment), [model.NumCluster, model.NumProtein])
    model.ClusterAssignment = sparse(model.IdxCluster(:), ...
        (1:model.NumProtein).', 1, model.NumCluster, model.NumProtein);
end
pooling = model.ClusterAssignment * ...
    spdiags(model.Aprob, 0, model.NumProtein, model.NumProtein);
heads = [model.Babt, model.Bmta, model.Bwmh];
splits = {'Train', 'Valid'};
if includeTest
    splits{end + 1} = 'Test';
else
    testFields = {'HTest', 'ZTest', 'LogitsTest', 'PabtTest', 'PmtaTest', 'PwmhTest'};
    staleFields = testFields(isfield(model, testFields));
    if ~isempty(staleFields)
        model = rmfield(model, staleFields);
    end
end

sampleCount = size(model.XTrain, 2) + size(model.XValid, 2);
if includeTest
    sampleCount = sampleCount + size(model.XTest, 2);
end
useTransform = sampleCount >= model.NumProtein;
if useTransform
    propagation = solver_prop(factor, full(model.Udiag));
end
for index = 1:numel(splits)
    split = splits{index};
    features = model.(['X', split]);
    if size(features, 1) ~= model.NumProtein || ~isreal(features) || ...
            any(~isfinite(features(:)))
        error('NeuroFANN:InvalidFeatures', ...
            'Each feature matrix must contain finite real values with one row per protein.');
    end
    if useTransform
        hidden = propagation * features;
        if any(~isfinite(hidden(:)))
            error('NeuroFANN:NonfinitePropagation', ...
                'Propagation produced nonfinite hidden features.');
        end
    else
        hidden = solver_prop(factor, bsxfun(@times, model.Uprot, features));
    end
    pooled = pooling * hidden;
    logits = heads.' * pooled;
    if any(~isfinite(logits(:)))
        error('NeuroFANN:NonfinitePropagation', ...
            'Forward propagation produced nonfinite logits.');
    end
    probabilities = zeros(size(logits), 'like', logits);
    positive = logits >= 0;
    probabilities(positive) = 1 ./ (1 + exp(-logits(positive)));
    negativeExp = exp(logits(~positive));
    probabilities(~positive) = negativeExp ./ (1 + negativeExp);
    model.(['H', split]) = hidden;
    model.(['Z', split]) = pooled;
    model.(['Logits', split]) = logits;
    model.(['Pabt', split]) = probabilities(1, :);
    model.(['Pmta', split]) = probabilities(2, :);
    model.(['Pwmh', split]) = probabilities(3, :);
end

end
