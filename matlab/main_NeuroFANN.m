function [ScoreABT, ScoreMTA, ScoreWMH, Results] = main_NeuroFANN(dataSource, overrides)

codeDirectory = fileparts(mfilename('fullpath'));
originalPath = path;
pathCleanup = onCleanup(@() path(originalPath));
addpath(codeDirectory);
if nargin < 1 || isempty(dataSource)
    dataSource = fullfile(codeDirectory, 'data_sample.mat');
end
if nargin < 2
    overrides = struct;
end
if isstruct(dataSource)
    dataset = dataSource;
    source = 'supplied Dataset struct';
elseif (ischar(dataSource) && isrow(dataSource)) || (isstring(dataSource) && isscalar(dataSource))
    if ~isfile(dataSource)
        error('NeuroFANN:MissingDataFile', 'Dataset file does not exist: %s.', dataSource);
    end
    loaded = load(dataSource, 'Dataset');
    if ~isfield(loaded, 'Dataset')
        error('NeuroFANN:InvalidDataset', 'The MAT file must contain a variable named Dataset.');
    end
    dataset = loaded.Dataset;
    source = char(dataSource);
else
    error('NeuroFANN:InvalidDataset', 'Provide a Dataset struct or MAT-file path.');
end
dataset = split_validset(dataset);
parameter = model_options(overrides, dataset);
ScoreABT = cell(parameter.NumIter, parameter.NumFold);
ScoreMTA = cell(parameter.NumIter, parameter.NumFold);
ScoreWMH = cell(parameter.NumIter, parameter.NumFold);
Results = struct;
if nargout >= 4
    Results.Parameter = parameter;
    Results.DataSource = source;
    Results.MATLABVersion = version;
    Results.OutOfFoldABT = nan(parameter.NumIter, size(dataset.XData,2));
    Results.OutOfFoldMTA = Results.OutOfFoldABT;
    Results.OutOfFoldWMH = Results.OutOfFoldABT;
    Results.Folds = cell(parameter.NumIter, parameter.NumFold);
end
timer = tic;
for iteration = 1:parameter.NumIter
    context = struct('IdxIter', iteration, 'Dataset', dataset, 'Parameter', parameter);
    for fold = 1:parameter.NumFold
        model = model_initialize(context, fold);
        model = param_initialize(model);
        model.AdamParam = init_adamopt(model.NumParam, model.LearnRate);
        model = param_training(model);
        ScoreABT{iteration,fold} = model.PabtTest;
        ScoreMTA{iteration,fold} = model.PmtaTest;
        ScoreWMH{iteration,fold} = model.PwmhTest;
        if nargout >= 4
            Results.OutOfFoldABT(iteration,model.IdxValid) = model.PabtValid;
            Results.OutOfFoldMTA(iteration,model.IdxValid) = model.PmtaValid;
            Results.OutOfFoldWMH(iteration,model.IdxValid) = model.PwmhValid;
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
            Results.Folds{iteration,fold} = summary;
        end
        if parameter.Verbose
            fprintf('Iteration %d/%d, fold %d/%d: epoch %d, validation loss %.8g\n', ...
                iteration, parameter.NumIter, fold, parameter.NumFold, ...
                model.BestEpoch, model.BestValidationLoss);
        end
    end
end
if nargout >= 4
    Results.ElapsedSeconds = toc(timer);
end

end