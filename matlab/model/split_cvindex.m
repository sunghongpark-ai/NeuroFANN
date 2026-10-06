function folds = split_cvindex(dataset, numIter, numFold, seed)

if nargin < 2 || isempty(numIter)
    numIter = 100;
end
if nargin < 3 || isempty(numFold)
    numFold = 5;
end
if nargin < 4 || isempty(seed)
    seed = 1;
end
dataset = split_validset(dataset, false);
sampleCount = size(dataset.XData, 2);
checkInteger(numIter, 'numIter', 1, Inf);
checkInteger(numFold, 'numFold', 2, sampleCount);
checkInteger(seed, 'seed', 0, 4294967295);
numIter = double(numIter);
numFold = double(numFold);
pattern = 4 * dataset.YabtData + 2 * dataset.YmtaData + dataset.YwmhData;
draws = rand_mt19937(seed, numIter * sampleCount);
assignment = mod(0:sampleCount - 1, numFold) + 1;
folds = zeros(numIter, sampleCount);
for iteration = 1:numIter
    keys = draws((iteration - 1) * sampleCount + (1:sampleCount));
    [~, order] = sortrows([pattern(:), keys(:)]);
    folds(iteration, order) = assignment;
end

end

function checkInteger(value, name, lower, upper)

if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ~isfinite(value) || ...
        value ~= fix(value) || value < lower || value > upper
    error('NeuroFANN:InvalidParameter', '%s must be an integer from %g to %g.', name, lower, upper);
end

end
