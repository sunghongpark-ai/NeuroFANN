function dataset = split_validset(dataset)

required = {'XData','XTest','YabtData','YmtaData','YwmhData','Lppi', ...
    'NumProtein','IdxCluster','NumCluster','CVindex'};
if ~isstruct(dataset) || ~isscalar(dataset)
    error('NeuroFANN:InvalidDataset', 'Dataset must be a scalar struct.');
end
missing = required(~isfield(dataset, required));
if ~isempty(missing)
    error('NeuroFANN:InvalidDataset', 'Dataset is missing: %s.', strjoin(missing, ', '));
end
validateCount(dataset.NumProtein, 'NumProtein');
validateCount(dataset.NumCluster, 'NumCluster');
dataset.NumProtein = double(dataset.NumProtein);
dataset.NumCluster = double(dataset.NumCluster);
proteinCount = dataset.NumProtein;
clusterCount = dataset.NumCluster;
if clusterCount > proteinCount
    error('NeuroFANN:InvalidDataset', 'NumCluster cannot exceed NumProtein.');
end
validateMatrix(dataset.XData, 'XData');
validateMatrix(dataset.XTest, 'XTest');
if size(dataset.XData,1) ~= proteinCount || size(dataset.XTest,1) ~= proteinCount || ...
        size(dataset.XData,2) < 2
    error('NeuroFANN:InvalidDataset', 'XData and XTest require NumProtein rows; XData needs at least two samples.');
end
dataset.XData = full(double(dataset.XData));
dataset.XTest = full(double(dataset.XTest));
sampleCount = size(dataset.XData,2);
testCount = size(dataset.XTest,2);
validateMatrix(dataset.Lppi, 'Lppi');
if ~isequal(size(dataset.Lppi), [proteinCount proteinCount])
    error('NeuroFANN:InvalidDataset', 'Lppi must be NumProtein-by-NumProtein.');
end
dataset.Lppi = double(dataset.Lppi);
clusters = dataset.IdxCluster;
if ~isnumeric(clusters) || ~isreal(clusters) || ~isvector(clusters) || ...
        numel(clusters) ~= proteinCount || any(~isfinite(clusters(:))) || ...
        any(clusters(:) ~= fix(clusters(:))) || any(clusters(:) < 1 | clusters(:) > clusterCount)
    error('NeuroFANN:InvalidDataset', 'IdxCluster must assign each protein an integer cluster in 1:NumCluster.');
end
dataset.IdxCluster = double(clusters(:));
if any(accumarray(dataset.IdxCluster, 1, [clusterCount 1]) == 0)
    error('NeuroFANN:InvalidDataset', 'Every cluster must contain at least one protein.');
end
targets = {'abt','mta','wmh'};
for index = 1:numel(targets)
    dataName = ['Y' targets{index} 'Data'];
    testName = ['Y' targets{index} 'Test'];
    dataset.(dataName) = validateLabels(dataset.(dataName), sampleCount, dataName);
    if ~isfield(dataset, testName) || isempty(dataset.(testName))
        dataset.(testName) = zeros(1,0);
    else
        dataset.(testName) = validateLabels(dataset.(testName), testCount, testName);
    end
end
folds = dataset.CVindex;
if ~isnumeric(folds) || ~isreal(folds) || ~ismatrix(folds) || isempty(folds) || ...
        size(folds,2) ~= sampleCount || any(~isfinite(folds(:))) || ...
        any(folds(:) ~= fix(folds(:))) || any(folds(:) < 1)
    error('NeuroFANN:InvalidDataset', 'CVindex must contain positive integer fold IDs with one column per XData sample.');
end
dataset.CVindex = double(folds);
if isfield(dataset, 'Wppi') && ~isempty(dataset.Wppi)
    validateMatrix(dataset.Wppi, 'Wppi');
    if ~isequal(size(dataset.Wppi), [proteinCount proteinCount])
        error('NeuroFANN:InvalidDataset', 'Wppi must be NumProtein-by-NumProtein when supplied.');
    end
    dataset.Wppi = double(dataset.Wppi);
else
    dataset.Wppi = [];
end
if ~isfield(dataset, 'IdxProtein') || isempty(dataset.IdxProtein)
    dataset.IdxProtein = string((1:proteinCount)');
elseif ~isvector(dataset.IdxProtein) || numel(dataset.IdxProtein) ~= proteinCount
    error('NeuroFANN:InvalidDataset', 'IdxProtein must contain one identifier per protein.');
end

end

function validateCount(value, name)

if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ...
        ~isfinite(value) || value < 1 || value ~= fix(value)
    error('NeuroFANN:InvalidDataset', '%s must be a positive integer scalar.', name);
end

end

function validateMatrix(value, name)

if ~(isa(value, 'double') || isa(value, 'single')) || ~isreal(value) || ...
        ~ismatrix(value) || any(~isfinite(value(:)))
    error('NeuroFANN:InvalidDataset', '%s must be a finite, real floating-point matrix.', name);
end

end

function labels = validateLabels(labels, count, name)

if ~(isnumeric(labels) || islogical(labels)) || ~isreal(labels) || ...
        ~isvector(labels) || numel(labels) ~= count || any(labels(:) ~= 0 & labels(:) ~= 1)
    error('NeuroFANN:InvalidDataset', '%s must contain one binary label per sample.', name);
end
labels = full(double(reshape(labels, 1, [])));

end