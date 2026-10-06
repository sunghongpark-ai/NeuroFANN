function dataset = split_validset(dataset, requireCV)

if nargin < 2 || isempty(requireCV)
    requireCV = true;
end
if ~isstruct(dataset) || ~isscalar(dataset)
    error('NeuroFANN:InvalidDataset', 'Dataset must be a scalar struct.');
end
required = {'XData', 'XTest', 'YabtData', 'YmtaData', 'YwmhData', ...
    'NumProtein', 'IdxCluster', 'NumCluster'};
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
if size(dataset.XData, 1) ~= proteinCount || size(dataset.XTest, 1) ~= proteinCount || ...
        size(dataset.XData, 2) < 2
    error('NeuroFANN:InvalidDataset', 'XData and XTest require NumProtein rows; XData needs at least two samples.');
end
dataset.XData = full(double(dataset.XData));
dataset.XTest = full(double(dataset.XTest));
sampleCount = size(dataset.XData, 2);
testCount = size(dataset.XTest, 2);
hasAdjacency = isfield(dataset, 'Wppi') && ~isempty(dataset.Wppi);
hasLaplacian = isfield(dataset, 'Lppi') && ~isempty(dataset.Lppi);
if hasAdjacency
    validateMatrix(dataset.Wppi, 'Wppi');
    if ~isequal(size(dataset.Wppi), [proteinCount proteinCount])
        error('NeuroFANN:InvalidDataset', 'Wppi must be NumProtein-by-NumProtein when supplied.');
    end
    dataset.Wppi = full(double(dataset.Wppi));
else
    dataset.Wppi = zeros(0, 0);
end
if hasLaplacian
    validateMatrix(dataset.Lppi, 'Lppi');
    if ~isequal(size(dataset.Lppi), [proteinCount proteinCount])
        error('NeuroFANN:InvalidDataset', 'Lppi must be NumProtein-by-NumProtein.');
    end
    dataset.Lppi = double(dataset.Lppi);
elseif hasAdjacency
    dataset.Lppi = ppi_laplacian(dataset.Wppi);
else
    error('NeuroFANN:InvalidDataset', 'Dataset requires Lppi or the PPI weights Wppi.');
end
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
targets = {'abt', 'mta', 'wmh'};
for index = 1:numel(targets)
    dataName = ['Y' targets{index} 'Data'];
    testName = ['Y' targets{index} 'Test'];
    dataset.(dataName) = validateLabels(dataset.(dataName), sampleCount, dataName);
    if ~isfield(dataset, testName) || isempty(dataset.(testName))
        dataset.(testName) = zeros(1, 0);
    else
        dataset.(testName) = validateLabels(dataset.(testName), testCount, testName);
    end
end
if ~isfield(dataset, 'CVindex') || isempty(dataset.CVindex)
    if requireCV
        error('NeuroFANN:InvalidDataset', ...
            'Dataset has no cross-validation folds (CVindex); create them with split_cvindex.');
    end
    dataset.CVindex = zeros(0, sampleCount);
else
    folds = dataset.CVindex;
    if ~isnumeric(folds) || ~isreal(folds) || ~ismatrix(folds) || ...
            size(folds, 2) ~= sampleCount || any(~isfinite(folds(:))) || ...
            any(folds(:) ~= fix(folds(:))) || any(folds(:) < 1)
        error('NeuroFANN:InvalidDataset', ...
            'CVindex must contain positive integer fold IDs with one column per XData sample.');
    end
    dataset.CVindex = double(folds);
end
if ~isfield(dataset, 'IdxProtein') || isempty(dataset.IdxProtein)
    dataset.IdxProtein = defaultNames('', proteinCount, 0);
else
    dataset.IdxProtein = textColumn(dataset.IdxProtein, proteinCount, 'IdxProtein');
end
if ~isfield(dataset, 'IdxSampleData') || isempty(dataset.IdxSampleData)
    dataset.IdxSampleData = defaultNames('D', sampleCount, 3);
else
    dataset.IdxSampleData = textColumn(dataset.IdxSampleData, sampleCount, 'IdxSampleData');
end
if ~isfield(dataset, 'IdxSampleTest') || isempty(dataset.IdxSampleTest)
    dataset.IdxSampleTest = defaultNames('V', testCount, 3);
else
    dataset.IdxSampleTest = textColumn(dataset.IdxSampleTest, testCount, 'IdxSampleTest');
end
if numel(unique(dataset.IdxProtein)) ~= proteinCount
    error('NeuroFANN:InvalidDataset', 'IdxProtein identifiers must be unique.');
end
if numel(unique([dataset.IdxSampleData; dataset.IdxSampleTest])) ~= sampleCount + testCount
    error('NeuroFANN:InvalidDataset', 'Sample identifiers must be unique across XData and XTest.');
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
        ~ismatrix(value) || any(~isfinite(nonzeros(value)))
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

function names = textColumn(value, count, name)

if ischar(value)
    names = cellstr(value);
elseif ~isvector(value)
    error('NeuroFANN:InvalidDataset', '%s must contain one identifier per entry.', name);
elseif iscellstr(value)
    names = value(:);
elseif isstring(value)
    names = cellstr(value(:));
elseif (isnumeric(value) || islogical(value)) && isreal(value) && all(isfinite(double(value(:))))
    names = arrayfun(@(entry) sprintf('%.17g', entry), double(value(:)), 'UniformOutput', false);
else
    error('NeuroFANN:InvalidDataset', '%s must contain text or finite numeric identifiers.', name);
end
if numel(names) ~= count
    error('NeuroFANN:InvalidDataset', '%s must contain one identifier per entry.', name);
end
names = reshape(names, [], 1);

end

function names = defaultNames(prefix, count, minimumWidth)

if minimumWidth > 0
    width = max(minimumWidth, numel(sprintf('%d', count)));
    pattern = sprintf('%%s%%0%dd', width);
else
    pattern = '%s%d';
end
names = arrayfun(@(entry) sprintf(pattern, prefix, entry), (1:count).', 'UniformOutput', false);

end
