function data_writecsv(dataset, file)

if ~((ischar(file) && isrow(file)) || (isstring(file) && isscalar(file)))
    error('NeuroFANN:InvalidDataset', 'Provide the CSV file path as text.');
end
file = char(file);
dataset = split_validset(dataset, false);
if isempty(dataset.Wppi)
    error('NeuroFANN:InvalidDataset', 'Writing a CSV dataset requires the PPI weights Wppi.');
end
if ~isequal(ppi_laplacian(dataset.Wppi), full(dataset.Lppi))
    error('NeuroFANN:InvalidDataset', ...
        'Lppi must equal the normalized Laplacian of Wppi because the CSV format stores Wppi only.');
end
reserved = {'record', 'id', 'split', 'y_abt', 'y_mta', 'y_wmh'};
proteinNames = dataset.IdxProtein;
for index = 1:numel(proteinNames)
    checkName(proteinNames{index}, 'Protein name');
    if any(strcmp(proteinNames{index}, reserved)) || ~isempty(regexp(proteinNames{index}, '^cv[0-9]+$', 'once'))
        error('NeuroFANN:InvalidDataset', 'Protein name "%s" is reserved by the CSV format.', proteinNames{index});
    end
end
sampleIds = [dataset.IdxSampleData; dataset.IdxSampleTest];
for index = 1:numel(sampleIds)
    checkName(sampleIds{index}, 'Sample id');
end
foldCount = size(dataset.CVindex, 1);
width = max(3, numel(sprintf('%d', foldCount)));
foldNames = arrayfun(@(index) sprintf(sprintf('cv%%0%dd', width), index), 1:foldCount, 'UniformOutput', false);
proteinCount = dataset.NumProtein;
labelData = [dataset.YabtData; dataset.YmtaData; dataset.YwmhData];
labelTest = {dataset.YabtTest, dataset.YmtaTest, dataset.YwmhTest};
emptyFolds = repmat({''}, 1, foldCount);
emptyLabels = repmat({''}, 1, 3);

lines = cell(2 + proteinCount + numel(sampleIds), 1);
lines{1} = strjoin([reserved, foldNames, reshape(proteinNames, 1, [])], ',');
lines{2} = strjoin([{'cluster', 'cluster', ''}, emptyLabels, emptyFolds, ...
    numberFields(dataset.IdxCluster)], ',');
for protein = 1:proteinCount
    lines{2 + protein} = strjoin([{'network', proteinNames{protein}, ''}, emptyLabels, emptyFolds, ...
        numberFields(dataset.Wppi(protein, :))], ',');
end
offset = 2 + proteinCount;
for sample = 1:size(dataset.XData, 2)
    lines{offset + sample} = strjoin([{'sample', dataset.IdxSampleData{sample}, 'train'}, ...
        numberFields(labelData(:, sample)), numberFields(dataset.CVindex(:, sample)), ...
        numberFields(dataset.XData(:, sample))], ',');
end
offset = offset + size(dataset.XData, 2);
for sample = 1:size(dataset.XTest, 2)
    labels = emptyLabels;
    for target = 1:3
        if ~isempty(labelTest{target})
            labels{target} = sprintf('%.17g', labelTest{target}(sample));
        end
    end
    lines{offset + sample} = strjoin([{'sample', dataset.IdxSampleTest{sample}, 'test'}, ...
        labels, emptyFolds, numberFields(dataset.XTest(:, sample))], ',');
end

[stream, message] = fopen(file, 'w');
if stream < 0
    error('NeuroFANN:OutputFailure', 'Cannot create %s: %s', file, message);
end
closer = onCleanup(@() fclose(stream));
fprintf(stream, '%s\n', lines{:});
clear closer

end

function fields = numberFields(values)

if isempty(values)
    fields = cell(1, 0);
    return
end
fields = regexp(sprintf('%.17g,', values), ',', 'split');
fields = fields(1:end - 1);

end

function checkName(name, label)

if ~ischar(name) || isempty(name) || any(name < 32 | name > 126) || any(name == ',') || ...
        any(name == '"') || isspace(name(1)) || isspace(name(end))
    error('NeuroFANN:InvalidDataset', '%s "%s" cannot be written as a plain CSV field.', label, char(name));
end

end
