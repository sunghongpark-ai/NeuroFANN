function parameter = model_options(overrides, dataset)

if nargin < 1 || isempty(overrides)
    overrides = struct;
end
if ~isstruct(overrides) || ~isscalar(overrides)
    error('NeuroFANN:InvalidParameter', 'Parameter overrides must be a scalar struct.');
end
parameter = struct('NumIter', size(dataset.CVindex,1), ...
    'NumFold', max(dataset.CVindex(1,:)), 'MaxEpoch', 500, ...
    'LearnRate', 0.001, 'RegCoeff', 0.005, 'Seed', 1, ...
    'StoreHistory', false, 'Verbose', false);
names = fieldnames(overrides);
for index = 1:numel(names)
    name = names{index};
    if ~isfield(parameter, name)
        error('NeuroFANN:InvalidParameter', 'Unknown parameter: %s.', name);
    end
    parameter.(name) = overrides.(name);
end
counts = {'NumIter','NumFold','MaxEpoch'};
for index = 1:numel(counts)
    name = counts{index};
    value = parameter.(name);
    if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ...
            ~isfinite(value) || value < 1 || value ~= fix(value)
        error('NeuroFANN:InvalidParameter', '%s must be a positive integer scalar.', name);
    end
    parameter.(name) = double(value);
end
if parameter.NumIter > size(dataset.CVindex,1) || parameter.NumFold < 2 || ...
        parameter.NumFold > size(dataset.CVindex,2)
    error('NeuroFANN:InvalidParameter', 'NumIter exceeds CVindex or NumFold is outside 2:NumSamples.');
end
for iteration = 1:parameter.NumIter
    actual = unique(dataset.CVindex(iteration,:));
    if ~isequal(actual(:)', 1:parameter.NumFold)
        error('NeuroFANN:InvalidFold', 'CVindex row %d must include exactly the fold IDs 1:NumFold.', iteration);
    end
end
scalars = {'LearnRate','RegCoeff','Seed'};
for index = 1:numel(scalars)
    name = scalars{index};
    value = parameter.(name);
    if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ~isfinite(value)
        error('NeuroFANN:InvalidParameter', '%s must be a finite real scalar.', name);
    end
    parameter.(name) = double(value);
end
if parameter.LearnRate <= 0 || parameter.RegCoeff < 0
    error('NeuroFANN:InvalidParameter', 'LearnRate must be positive and RegCoeff nonnegative.');
end
if parameter.Seed < 0 || parameter.Seed ~= fix(parameter.Seed) || ...
        parameter.Seed + parameter.NumIter - 1 > double(intmax('uint32'))
    error('NeuroFANN:InvalidParameter', 'Seed through Seed+NumIter-1 must be integers in the uint32 range.');
end
flags = {'StoreHistory','Verbose'};
for index = 1:numel(flags)
    name = flags{index};
    value = parameter.(name);
    if ~(islogical(value) || isnumeric(value)) || ~isreal(value) || ~isscalar(value) || ...
            ~(value == 0 || value == 1)
        error('NeuroFANN:InvalidParameter', '%s must be a scalar logical flag.', name);
    end
    parameter.(name) = logical(value);
end

end