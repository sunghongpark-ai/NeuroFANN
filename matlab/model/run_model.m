function [ScoreABT, ScoreMTA, ScoreWMH, Results] = run_model(dataSource, overrides)

codeDirectory = fileparts(mfilename('fullpath'));
originalPath = path;
pathCleanup = onCleanup(@() path(originalPath));
addpath(codeDirectory);
if nargin < 1 || isempty(dataSource)
    dataSource = fullfile(fileparts(fileparts(codeDirectory)), 'dataset', 'sample.csv');
end
if nargin < 2 || isempty(overrides)
    overrides = struct;
end
[dataset, source] = loadDataset(dataSource);
dataset = split_validset(dataset);
parameter = model_options(overrides, dataset);
iterations = parameter.NumIter;
folds = parameter.NumFold;
sampleCount = size(dataset.XData, 2);
ScoreABT = cell(iterations, folds);
ScoreMTA = cell(iterations, folds);
ScoreWMH = cell(iterations, folds);
Results = struct;
Results.Parameter = parameter;
Results.DataSource = source;
Results.Runtime = runtimeName();
Results.OutOfFoldABT = nan(iterations, sampleCount);
Results.OutOfFoldMTA = nan(iterations, sampleCount);
Results.OutOfFoldWMH = nan(iterations, sampleCount);
Results.Folds = cell(iterations, folds);
timer = tic;
for iteration = 1:iterations
    context = struct('IdxIter', iteration, 'Dataset', dataset, 'Parameter', parameter);
    for fold = 1:folds
        model = param_initialize(model_initialize(context, fold));
        model.AdamParam = init_adamopt(model.NumParam, model.LearnRate);
        model = param_training(model);
        ScoreABT{iteration, fold} = model.PabtTest;
        ScoreMTA{iteration, fold} = model.PmtaTest;
        ScoreWMH{iteration, fold} = model.PwmhTest;
        Results.OutOfFoldABT(iteration, model.IdxValid) = model.PabtValid;
        Results.OutOfFoldMTA(iteration, model.IdxValid) = model.PmtaValid;
        Results.OutOfFoldWMH(iteration, model.IdxValid) = model.PwmhValid;
        summary = struct('BestEpoch', model.BestEpoch, ...
            'BestValidationLoss', model.BestValidationLoss, ...
            'WeightParam', model.WeightParam, 'SizeParam', model.SizeParam, ...
            'AdamParam', model.AdamParam, 'LossTrain', model.LossTrain, ...
            'LossValid', model.LossValid, 'ObjectiveTrain', model.ObjectiveTrain, ...
            'IdxTrain', model.IdxTrain, 'IdxValid', model.IdxValid, ...
            'Seed', model.Seed, 'TrainingTime', model.TrainingTime);
        if parameter.StoreHistory
            summary.WeightEpoch = model.WeightEpoch;
        end
        Results.Folds{iteration, fold} = summary;
        if parameter.Verbose
            fprintf('Iteration %d/%d, fold %d/%d: epoch %d, validation loss %.8g\n', ...
                iteration, iterations, fold, folds, model.BestEpoch, model.BestValidationLoss);
        end
    end
end
Results.ElapsedSeconds = toc(timer);
Results.Evaluation = model_evaluate(ScoreABT, ScoreMTA, ScoreWMH, Results, dataset);
if parameter.Verbose
    evaluation = Results.Evaluation;
    fprintf('Test AUROC, mean of %d models: ABT %.4f, MTA %.4f, WMH %.4f\n', ...
        iterations * folds, evaluation.TestAUROCMean);
    fprintf('Test AUROC, ensemble mean risk: ABT %.4f, MTA %.4f, WMH %.4f\n', ...
        evaluation.EnsembleTestAUROC);
    fprintf('Out-of-fold AUROC, mean of %d iterations: ABT %.4f, MTA %.4f, WMH %.4f\n', ...
        iterations, evaluation.OutOfFoldAUROCMean);
end

end

function [dataset, source] = loadDataset(dataSource)

if isstruct(dataSource)
    dataset = dataSource;
    source = 'supplied Dataset struct';
    return
end
if ~((ischar(dataSource) && isrow(dataSource)) || (isstring(dataSource) && isscalar(dataSource)))
    error('NeuroFANN:InvalidDataset', 'Provide a Dataset struct, a CSV file path, or a MAT-file path.');
end
source = char(dataSource);
[~, ~, extension] = fileparts(source);
switch lower(extension)
    case '.csv'
        dataset = data_readcsv(source);
    case '.mat'
        if ~isfile(source)
            error('NeuroFANN:MissingDataFile', 'Dataset file does not exist: %s.', source);
        end
        try
            variables = whos('-file', source);
        catch failure
            error('NeuroFANN:InvalidDataset', 'Cannot read %s: %s', source, failure.message);
        end
        if ~any(strcmp({variables.name}, 'Dataset'))
            error('NeuroFANN:InvalidDataset', 'The MAT file must contain a variable named Dataset.');
        end
        loaded = load(source, 'Dataset');
        dataset = loaded.Dataset;
    otherwise
        error('NeuroFANN:UnsupportedDataFile', 'Use a .csv or .mat dataset file: %s.', source);
end

end

function name = runtimeName

if exist('OCTAVE_VERSION', 'builtin') == 5
    name = ['GNU Octave ' version];
else
    name = ['MATLAB ' version];
end

end
