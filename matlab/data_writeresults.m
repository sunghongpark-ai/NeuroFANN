function files = data_writeresults(Results, outputDirectory)

if ~isstruct(Results) || ~isscalar(Results) || ~isfield(Results, 'Evaluation') || ~isfield(Results, 'Parameter')
    error('NeuroFANN:InvalidResults', 'Results must be the fourth output of main_NeuroFANN.');
end
if ~((ischar(outputDirectory) && isrow(outputDirectory)) || (isstring(outputDirectory) && isscalar(outputDirectory)))
    error('NeuroFANN:InvalidResults', 'Provide the output folder as text.');
end
outputDirectory = char(outputDirectory);
if ~isfolder(outputDirectory)
    [created, message] = mkdir(outputDirectory);
    if ~created
        error('NeuroFANN:OutputFailure', 'Cannot create %s: %s', outputDirectory, message);
    end
end
evaluation = Results.Evaluation;
targets = {'abt', 'mta', 'wmh'};
files = cell(5, 1);

labelHeader = strcat('y_', targets);
riskHeader = strcat('risk_', targets);
files{1} = fullfile(outputDirectory, 'predictions_test.csv');
writeTable(files{1}, [{'id'}, labelHeader, riskHeader], ...
    [evaluation.IdxSampleTest(:), numberCells(evaluation.LabelsTest.'), ...
    numberCells(evaluation.EnsembleTestScore.')]);
files{2} = fullfile(outputDirectory, 'predictions_oof.csv');
writeTable(files{2}, [{'id'}, labelHeader, riskHeader], ...
    [evaluation.IdxSampleData(:), numberCells(evaluation.LabelsData.'), ...
    numberCells(evaluation.MeanOutOfFoldScore.')]);

[iterations, folds, ~] = size(evaluation.TestAUROC);
[foldIndex, iterationIndex] = meshgrid(1:folds, 1:iterations);
iterationIndex = reshape(iterationIndex.', [], 1);
foldIndex = reshape(foldIndex.', [], 1);
modelValues = zeros(iterations * folds, 3);
for target = 1:3
    modelValues(:, target) = reshape(evaluation.TestAUROC(:, :, target).', [], 1);
end
files{3} = fullfile(outputDirectory, 'auroc_models.csv');
writeTable(files{3}, [{'iteration', 'fold'}, strcat('auroc_', targets)], ...
    [numberCells(iterationIndex), numberCells(foldIndex), numberCells(modelValues)]);

summary = [evaluation.TestAUROCMean; evaluation.TestAUROCStd; evaluation.TestAUROCMin; ...
    evaluation.TestAUROCMax; evaluation.EnsembleTestAUROC; evaluation.OutOfFoldAUROCMean; ...
    evaluation.OutOfFoldAUROCStd; evaluation.OutOfFoldAUROCMin; evaluation.OutOfFoldAUROCMax].';
files{4} = fullfile(outputDirectory, 'auroc_summary.csv');
writeTable(files{4}, {'target', 'test_mean', 'test_std', 'test_min', 'test_max', 'test_ensemble', ...
    'oof_mean', 'oof_std', 'oof_min', 'oof_max'}, [upper(targets(:)), numberCells(summary)]);

report = struct;
report.DataSource = Results.DataSource;
report.Runtime = Results.Runtime;
report.Parameter = Results.Parameter;
report.NumModels = evaluation.NumModels;
report.NumTrain = numel(evaluation.IdxSampleData);
report.NumTest = numel(evaluation.IdxSampleTest);
report.ElapsedSeconds = Results.ElapsedSeconds;
report.Targets = evaluation.Targets;
report.TestAUROCMean = evaluation.TestAUROCMean;
report.TestAUROCStd = evaluation.TestAUROCStd;
report.TestAUROCOverall = evaluation.TestAUROCOverall;
report.EnsembleTestAUROC = evaluation.EnsembleTestAUROC;
report.OutOfFoldAUROCMean = evaluation.OutOfFoldAUROCMean;
report.OutOfFoldAUROCStd = evaluation.OutOfFoldAUROCStd;
serialized = '';
if exist('OCTAVE_VERSION', 'builtin') ~= 5
    try
        serialized = jsonencode(report, 'PrettyPrint', true);
    catch
        serialized = '';
    end
end
if isempty(serialized)
    serialized = jsonencode(report);
end
files{5} = fullfile(outputDirectory, 'run_summary.json');
writeLines(files{5}, {serialized});

end

function cells = numberCells(values)

cells = cell(size(values));
for index = 1:numel(values)
    if isnan(values(index))
        cells{index} = '';
    else
        cells{index} = sprintf('%.17g', values(index));
    end
end

end

function writeTable(file, header, rows)

lines = cell(size(rows, 1) + 1, 1);
lines{1} = strjoin(header, ',');
for row = 1:size(rows, 1)
    lines{row + 1} = strjoin(rows(row, :), ',');
end
writeLines(file, lines);

end

function writeLines(file, lines)

[stream, message] = fopen(file, 'w', 'n', 'UTF-8');
if stream < 0
    error('NeuroFANN:OutputFailure', 'Cannot create %s: %s', file, message);
end
closer = onCleanup(@() fclose(stream));
fprintf(stream, '%s\n', lines{:});
clear closer

end
