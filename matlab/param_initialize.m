function model = param_initialize(model)

validateattributes(model.NumProtein, {'numeric'}, {'scalar', 'real', 'finite', 'integer', 'positive'}, mfilename, 'NumProtein');
validateattributes(model.NumCluster, {'numeric'}, {'scalar', 'real', 'finite', 'integer', 'positive'}, mfilename, 'NumCluster');
validateattributes(model.MaxEpoch, {'numeric'}, {'scalar', 'real', 'finite', 'integer', 'positive'}, mfilename, 'MaxEpoch');
model.NumProtein = double(model.NumProtein);
model.NumCluster = double(model.NumCluster);
model.MaxEpoch = double(model.MaxEpoch);

if ~isfield(model, 'Seed') || isempty(model.Seed)
    model.Seed = model.IdxIter;
end
validateattributes(model.Seed, {'numeric'}, {'scalar', 'real', 'finite', 'integer', 'nonnegative', '<=', double(intmax('uint32'))}, mfilename, 'Seed');
if ~isfield(model, 'StoreHistory') || isempty(model.StoreHistory)
    model.StoreHistory = false;
end
validateattributes(model.StoreHistory, {'numeric', 'logical'}, {'scalar', 'real', 'finite', 'binary'}, mfilename, 'StoreHistory');
model.StoreHistory = logical(model.StoreHistory);

stream = RandStream('mt19937ar', 'Seed', double(model.Seed));
model.Uprot = ones(model.NumProtein, 1);
model.Aclus = zeros(model.NumProtein, 1);
scale = sqrt(6 / (model.NumCluster + 1));
model.Babt = (2 * rand(stream, model.NumCluster, 1) - 1) * scale;
model.Bmta = (2 * rand(stream, model.NumCluster, 1) - 1) * scale;
model.Bwmh = (2 * rand(stream, model.NumCluster, 1) - 1) * scale;

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
