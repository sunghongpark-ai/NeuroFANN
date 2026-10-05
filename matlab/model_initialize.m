function init = model_initialize(model, fold)

if ~isstruct(model) || ~isscalar(model) || ~all(isfield(model, {'Dataset','Parameter','IdxIter'}))
    error('NeuroFANN:InvalidDataset', 'Model must contain Dataset, Parameter, and IdxIter.');
end
dataset = split_validset(model.Dataset);
parameter = model_options(model.Parameter, dataset);
if ~isnumeric(model.IdxIter) || ~isreal(model.IdxIter) || ~isscalar(model.IdxIter) || ~isfinite(model.IdxIter) || ...
        model.IdxIter ~= fix(model.IdxIter) || model.IdxIter < 1 || model.IdxIter > parameter.NumIter
    error('NeuroFANN:InvalidParameter', 'IdxIter must identify an available cross-validation iteration.');
end
if ~isnumeric(fold) || ~isreal(fold) || ~isscalar(fold) || ~isfinite(fold) || fold ~= fix(fold) || ...
        fold < 1 || fold > parameter.NumFold
    error('NeuroFANN:InvalidFold', 'Fold must be an integer between 1 and NumFold.');
end
init.IdxIter = double(model.IdxIter);
init.IdxFold = double(fold);
init.NumFold = parameter.NumFold;
init.IdxTrain = find(dataset.CVindex(model.IdxIter,:) ~= fold);
init.IdxValid = find(dataset.CVindex(model.IdxIter,:) == fold);
init.NumTrain = numel(init.IdxTrain);
init.NumValid = numel(init.IdxValid);
init.NumTest = size(dataset.XTest, 2);
init.XTrain = dataset.XData(:,init.IdxTrain);
init.XValid = dataset.XData(:,init.IdxValid);
init.XTest = dataset.XTest;
targets = {'abt','mta','wmh'};
for index = 1:numel(targets)
    prefix = ['Y' targets{index}];
    labels = dataset.([prefix 'Data']);
    init.([prefix 'Train']) = labels(init.IdxTrain);
    init.([prefix 'Valid']) = labels(init.IdxValid);
    init.([prefix 'Test']) = dataset.([prefix 'Test']);
end
init.Wppi = dataset.Wppi;
init.Lppi = dataset.Lppi;
init.IdxProtein = dataset.IdxProtein;
init.NumProtein = dataset.NumProtein;
init.IdxCluster = dataset.IdxCluster;
init.NumCluster = dataset.NumCluster;
init.ClusterMembers = cell(init.NumCluster,1);
for cluster = 1:init.NumCluster
    init.ClusterMembers{cluster} = find(init.IdxCluster == cluster);
end
init.ClusterAssignment = sparse(init.IdxCluster, (1:init.NumProtein)', 1, ...
    init.NumCluster, init.NumProtein);
init.MaxEpoch = parameter.MaxEpoch;
init.LearnRate = parameter.LearnRate;
init.RegCoeff = parameter.RegCoeff;
init.Seed = parameter.Seed + init.IdxIter - 1;
init.StoreHistory = parameter.StoreHistory;

end
