function dataset = data_readcsv(file)

if ~((ischar(file) && isrow(file)) || (isstring(file) && isscalar(file)))
    error('NeuroFANN:InvalidDataset', 'Provide the CSV file path as text.');
end
file = char(file);
if ~isfile(file)
    error('NeuroFANN:MissingDataFile', 'Dataset file does not exist: %s.', file);
end
[stream, message] = fopen(file, 'r');
if stream < 0
    error('NeuroFANN:MissingDataFile', 'Cannot open %s: %s', file, message);
end
closer = onCleanup(@() fclose(stream));
bytes = reshape(fread(stream, Inf, '*uint8'), 1, []);
clear closer
if numel(bytes) >= 3 && isequal(bytes(1:3), uint8([239 187 191]))
    bytes = bytes(4:end);
end
if any(bytes > 126 | (bytes < 32 & bytes ~= 9 & bytes ~= 10 & bytes ~= 13))
    error('NeuroFANN:InvalidCSV', 'The CSV file must contain printable ASCII text only.');
end
text = char(bytes);
if any(text == '"')
    error('NeuroFANN:InvalidCSV', 'Quoted CSV fields are not supported.');
end
text(text == char(13)) = [];
padded = any(text == ' ' | text == char(9));
lines = regexp(text, '\n', 'split');
lineNumbers = 1:numel(lines);
if padded
    blank = cellfun('isempty', strtrim(lines));
else
    blank = cellfun('isempty', lines);
end
lines = lines(~blank);
lineNumbers = lineNumbers(~blank);
if numel(lines) < 2
    error('NeuroFANN:InvalidCSV', 'The CSV file needs a header and at least one record.');
end
pieces = regexp(lines, ',', 'split');
header = pieces{1};
columnCount = numel(header);
rowCount = numel(lines) - 1;
rowLines = lineNumbers(2:end);
fieldCounts = cellfun('length', pieces(2:end));
mismatch = find(fieldCounts ~= columnCount, 1);
if ~isempty(mismatch)
    error('NeuroFANN:InvalidCSV', 'Line %d has %d fields; the header has %d.', ...
        rowLines(mismatch), fieldCounts(mismatch), columnCount);
end
cells = reshape([pieces{2:end}], columnCount, rowCount).';
if padded
    header = strtrim(header);
    cells = strtrim(cells);
end

if any(cellfun('isempty', header))
    error('NeuroFANN:InvalidCSV', 'Every header field must have a name.');
end
if numel(unique(header)) ~= columnCount
    error('NeuroFANN:InvalidCSV', 'Header names must be unique.');
end
reserved = {'record', 'id', 'split', 'y_abt', 'y_mta', 'y_wmh'};
[present, location] = ismember(reserved, header);
if ~all(present)
    error('NeuroFANN:InvalidCSV', 'The header is missing: %s.', strjoin(reserved(~present), ', '));
end
recordColumn = location(1);
idColumn = location(2);
splitColumn = location(3);
labelColumns = location(4:6);
tokens = regexp(header, '^cv([0-9]+)$', 'tokens', 'once');
isFold = ~cellfun('isempty', tokens);
foldColumns = find(isFold);
foldNumbers = cellfun(@(token) str2double(token{1}), tokens(isFold));
[foldNumbers, order] = sort(foldNumbers);
foldColumns = foldColumns(order);
if ~isequal(foldNumbers, 1:numel(foldNumbers))
    error('NeuroFANN:InvalidCSV', 'Cross-validation columns must be cv1 to cvK without gaps or duplicates.');
end
proteinColumns = find(~ismember(header, reserved) & ~isFold);
if isempty(proteinColumns)
    error('NeuroFANN:InvalidCSV', 'The header must contain at least one protein column.');
end
proteinNames = reshape(header(proteinColumns), [], 1);
proteinCount = numel(proteinColumns);
auxiliaryColumns = [splitColumn, labelColumns, foldColumns];

records = cells(:, recordColumn);
known = ismember(records, {'cluster', 'network', 'sample'});
if ~all(known)
    row = find(~known, 1);
    error('NeuroFANN:InvalidCSV', 'Line %d has an unknown record type "%s".', rowLines(row), records{row});
end
clusterRows = find(strcmp(records, 'cluster'));
networkRows = find(strcmp(records, 'network'));
sampleRows = find(strcmp(records, 'sample'));
requireEmpty(cells, [clusterRows; networkRows], auxiliaryColumns, rowLines, header);

if numel(clusterRows) ~= 1
    error('NeuroFANN:InvalidCSV', 'The CSV file must contain exactly one cluster record.');
end
clusterIndex = parseNumbers(cells(clusterRows, proteinColumns), rowLines(clusterRows), proteinNames, false);
clusterIndex = clusterIndex(:);
if any(clusterIndex < 1 | clusterIndex ~= fix(clusterIndex))
    error('NeuroFANN:InvalidCSV', 'Line %d: cluster IDs must be positive integers.', rowLines(clusterRows));
end
clusterCount = max(clusterIndex);
if clusterCount > proteinCount || any(accumarray(clusterIndex, 1, [clusterCount 1]) == 0)
    error('NeuroFANN:InvalidCSV', 'Cluster IDs must use every integer from 1 to the largest ID.');
end

networkIds = cells(networkRows, idColumn);
[listed, position] = ismember(networkIds, proteinNames);
if numel(networkRows) ~= proteinCount || ~all(listed) || numel(unique(position)) ~= proteinCount
    error('NeuroFANN:InvalidCSV', ...
        'The CSV file needs exactly one network record per protein column, identified by id.');
end
weights = zeros(proteinCount);
weights(position, :) = parseNumbers(cells(networkRows, proteinColumns), rowLines(networkRows), proteinNames, false);
if any(weights(:) < 0)
    error('NeuroFANN:InvalidCSV', 'Network weights must be nonnegative.');
end
if ~isequal(weights, weights.')
    error('NeuroFANN:InvalidCSV', 'Network weights must form an exactly symmetric matrix.');
end

if isempty(sampleRows)
    error('NeuroFANN:InvalidCSV', 'The CSV file contains no sample records.');
end
sampleIds = cells(sampleRows, idColumn);
if any(cellfun('isempty', sampleIds)) || numel(unique(sampleIds)) ~= numel(sampleIds)
    error('NeuroFANN:InvalidCSV', 'Sample records need unique, nonempty ids.');
end
roles = cells(sampleRows, splitColumn);
isTrain = strcmp(roles, 'train');
isTest = strcmp(roles, 'test');
if ~all(isTrain | isTest)
    row = sampleRows(find(~(isTrain | isTest), 1));
    error('NeuroFANN:InvalidCSV', 'Line %d: split must be train or test.', rowLines(row));
end
if sum(isTrain) < 2
    error('NeuroFANN:InvalidCSV', 'At least two train samples are required.');
end
features = parseNumbers(cells(sampleRows, proteinColumns), rowLines(sampleRows), proteinNames, false);
labelNames = {'YabtData', 'YmtaData', 'YwmhData'; 'YabtTest', 'YmtaTest', 'YwmhTest'};
labelValues = cell(2, 3);
for target = 1:3
    values = parseNumbers(cells(sampleRows, labelColumns(target)), rowLines(sampleRows), ...
        header(labelColumns(target)), true);
    trainValues = values(isTrain);
    testValues = values(isTest);
    if any(isnan(trainValues)) || any(trainValues ~= 0 & trainValues ~= 1)
        error('NeuroFANN:InvalidCSV', 'Column %s needs a 0/1 label for every train sample.', ...
            header{labelColumns(target)});
    end
    if all(isnan(testValues))
        testValues = zeros(0, 1);
    elseif any(isnan(testValues)) || any(testValues ~= 0 & testValues ~= 1)
        error('NeuroFANN:InvalidCSV', 'Column %s needs 0/1 labels for all test samples or for none.', ...
            header{labelColumns(target)});
    end
    labelValues{1, target} = reshape(trainValues, 1, []);
    labelValues{2, target} = reshape(testValues, 1, []);
end
requireEmpty(cells, sampleRows(isTest), foldColumns, rowLines, header);
folds = parseNumbers(cells(sampleRows(isTrain), foldColumns), rowLines(sampleRows(isTrain)), ...
    header(foldColumns), false);
if any(folds(:) < 1 | folds(:) ~= fix(folds(:)))
    error('NeuroFANN:InvalidCSV', 'Fold IDs in cv columns must be positive integers.');
end

dataset = struct;
dataset.XData = features(isTrain, :).';
dataset.XTest = features(isTest, :).';
for target = 1:3
    dataset.(labelNames{1, target}) = labelValues{1, target};
end
for target = 1:3
    dataset.(labelNames{2, target}) = labelValues{2, target};
end
dataset.Wppi = weights;
dataset.Lppi = ppi_laplacian(weights);
dataset.IdxProtein = proteinNames;
dataset.NumProtein = proteinCount;
dataset.IdxCluster = clusterIndex;
dataset.NumCluster = clusterCount;
dataset.CVindex = folds.';
dataset.IdxSampleData = reshape(sampleIds(isTrain), [], 1);
dataset.IdxSampleTest = reshape(sampleIds(isTest), [], 1);
dataset = split_validset(dataset, false);

end

function values = parseNumbers(texts, lines, names, allowEmpty)

values = zeros(size(texts));
if isempty(texts)
    return
end
pattern = '^[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?$';
empty = cellfun('isempty', texts);
candidates = texts(~empty);
valid = allowEmpty || ~any(empty(:));
if valid && ~isempty(candidates)
    joined = sprintf('%s\n', candidates{:});
    valid = numel(regexp(joined, pattern, 'lineanchors', 'start')) == numel(candidates);
end
if ~valid
    invalid = cellfun('isempty', regexp(texts, pattern, 'once'));
    if allowEmpty
        invalid = invalid & ~empty;
    end
    [row, column] = find(invalid, 1);
    error('NeuroFANN:InvalidCSV', 'Line %d, column %s: "%s" is not a finite decimal number.', ...
        lines(row), names{column}, texts{row, column});
end
values = str2double(texts);
overflow = ~empty & ~isfinite(values);
if any(overflow(:))
    [row, column] = find(overflow, 1);
    error('NeuroFANN:InvalidCSV', 'Line %d, column %s: the value overflows double precision.', ...
        lines(row), names{column});
end
values(empty) = NaN;

end

function requireEmpty(cells, rows, columns, lines, header)

if isempty(rows) || isempty(columns)
    return
end
filled = ~cellfun('isempty', cells(rows, columns));
if any(filled(:))
    [row, column] = find(filled, 1);
    error('NeuroFANN:InvalidCSV', 'Line %d: column %s must be empty for this record.', ...
        lines(rows(row)), header{columns(column)});
end

end
