function model = model_forward(model, includeTest)

if nargin < 2
    includeTest = true;
end
if ~(isnumeric(includeTest) || islogical(includeTest)) || ~isreal(includeTest) || ...
        ~isscalar(includeTest) || ~(includeTest == 0 || includeTest == 1)
    error('NeuroFANN:InvalidParameter', 'includeTest must be a scalar logical flag.');
end

factor = factor_prop(model.Udiag + model.Lppi);
model.PropagationFactor = factor;
model.PropagationReciprocalCondition = factor.ReciprocalCondition;

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

featureSets = cell(1, numel(splits));
for index = 1:numel(splits)
    features = model.(['X', splits{index}]);
    if ~isfloat(features) || ~isreal(features) || ~ismatrix(features) || ...
            size(features, 1) ~= model.NumProtein || any(~isfinite(features(:)))
        error('NeuroFANN:InvalidFeatures', ...
            'Each feature matrix must contain finite real values with one row per protein.');
    end
    featureSets{index} = full(double(features));
end
useTransform = sum(cellfun('size', featureSets, 2)) >= model.NumProtein;
if useTransform
    propagation = solver_prop(factor, full(model.Udiag));
end
for index = 1:numel(splits)
    split = splits{index};
    features = featureSets{index};
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
    probabilities = zeros(size(logits));
    positive = logits >= 0;
    probabilities(positive) = 1 ./ (1 + exp(-logits(positive)));
    negativeExp = exp(logits(~positive));
    probabilities(~positive) = negativeExp ./ (1 + negativeExp);
    model.(['H', split]) = hidden;
    model.(['Z', split]) = full(pooled);
    model.(['Logits', split]) = full(logits);
    model.(['Pabt', split]) = full(probabilities(1, :));
    model.(['Pmta', split]) = full(probabilities(2, :));
    model.(['Pwmh', split]) = full(probabilities(3, :));
end

end
