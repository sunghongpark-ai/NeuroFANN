function model = param_initialize(model)

model.NumProtein = integerScalar(model.NumProtein, 'NumProtein', 1, Inf);
model.NumCluster = integerScalar(model.NumCluster, 'NumCluster', 1, Inf);
model.MaxEpoch = integerScalar(model.MaxEpoch, 'MaxEpoch', 1, Inf);
if ~isfield(model, 'Seed') || isempty(model.Seed)
    model.Seed = model.IdxIter;
end
model.Seed = integerScalar(model.Seed, 'Seed', 0, 4294967295);
if ~isfield(model, 'StoreHistory') || isempty(model.StoreHistory)
    model.StoreHistory = false;
end
flag = model.StoreHistory;
if ~(isnumeric(flag) || islogical(flag)) || ~isreal(flag) || ~isscalar(flag) || ~(flag == 0 || flag == 1)
    error('NeuroFANN:InvalidParameter', 'StoreHistory must be a scalar logical flag.');
end
model.StoreHistory = logical(flag);

clusters = model.NumCluster;
draws = rand_mt19937(model.Seed, 3 * clusters);
scale = sqrt(6 / (clusters + 1));
model.Uprot = ones(model.NumProtein, 1);
model.Aclus = zeros(model.NumProtein, 1);
model.Babt = (2 * draws(1:clusters) - 1) * scale;
model.Bmta = (2 * draws(clusters + (1:clusters)) - 1) * scale;
model.Bwmh = (2 * draws(2 * clusters + (1:clusters)) - 1) * scale;

model.SizeParam = [size(model.Uprot); size(model.Aclus); size(model.Babt); size(model.Bmta); size(model.Bwmh)];
model.NumParam = sum(prod(model.SizeParam, 2));
model.WeightParam = [model.Uprot; model.Aclus; model.Babt; model.Bmta; model.Bwmh];
if model.StoreHistory
    model.WeightEpoch = cell(model.MaxEpoch, 1);
else
    model.WeightEpoch = cell(0, 1);
end
model.LossTrain = nan(model.MaxEpoch, 1);
model.LossValid = nan(model.MaxEpoch, 1);
model.ObjectiveTrain = nan(model.MaxEpoch, 1);

end

function value = integerScalar(value, name, lower, upper)

if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ~isfinite(value) || ...
        value ~= fix(value) || value < lower || value > upper
    error('NeuroFANN:InvalidParameter', '%s must be an integer from %g to %g.', name, lower, upper);
end
value = double(value);

end
