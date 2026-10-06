function evaluation = model_evaluate(ScoreABT, ScoreMTA, ScoreWMH, Results, dataset)

dataset = split_validset(dataset, false);
testCount = size(dataset.XTest, 2);
sampleCount = size(dataset.XData, 2);
scoreSets = {ScoreABT, ScoreMTA, ScoreWMH};
[iterations, folds] = size(ScoreABT);
for target = 1:3
    scores = scoreSets{target};
    if ~iscell(scores) || ~isequal(size(scores), [iterations, folds]) || iterations < 1 || folds < 1
        error('NeuroFANN:InvalidScores', 'Score sets must be nonempty cell arrays of equal size.');
    end
    for model = 1:numel(scores)
        value = scores{model};
        if ~isnumeric(value) || ~isreal(value) || numel(value) ~= testCount || ...
                any(~isfinite(value(:))) || any(value(:) < 0 | value(:) > 1)
            error('NeuroFANN:InvalidScores', 'Each test score vector must hold one probability per test sample.');
        end
    end
end
outOfFold = {Results.OutOfFoldABT, Results.OutOfFoldMTA, Results.OutOfFoldWMH};
for target = 1:3
    if ~isnumeric(outOfFold{target}) || ~isreal(outOfFold{target}) || ...
            ~isequal(size(outOfFold{target}), [iterations, sampleCount])
        error('NeuroFANN:InvalidScores', 'Out-of-fold scores must be NumIter-by-NumData matrices.');
    end
end
labelsData = [dataset.YabtData; dataset.YmtaData; dataset.YwmhData];
labelsTest = nan(3, testCount);
testAvailable = [~isempty(dataset.YabtTest), ~isempty(dataset.YmtaTest), ~isempty(dataset.YwmhTest)];
testLabels = {dataset.YabtTest, dataset.YmtaTest, dataset.YwmhTest};
for target = 1:3
    if testAvailable(target)
        labelsTest(target, :) = testLabels{target};
    end
end

evaluation = struct;
evaluation.Targets = {'ABT', 'MTA', 'WMH'};
evaluation.NumModels = iterations * folds;
evaluation.IdxSampleData = dataset.IdxSampleData;
evaluation.IdxSampleTest = dataset.IdxSampleTest;
evaluation.LabelsData = labelsData;
evaluation.LabelsTest = labelsTest;
evaluation.TestAUROC = nan(iterations, folds, 3);
evaluation.EnsembleTestScore = zeros(3, testCount);
evaluation.EnsembleTestAUROC = nan(1, 3);
evaluation.OutOfFoldAUROC = nan(iterations, 3);
evaluation.MeanOutOfFoldScore = zeros(3, sampleCount);
for target = 1:3
    scores = scoreSets{target};
    total = zeros(1, testCount);
    for iteration = 1:iterations
        for fold = 1:folds
            current = reshape(double(scores{iteration, fold}), 1, []);
            total = total + current;
            if testAvailable(target)
                evaluation.TestAUROC(iteration, fold, target) = metric_auroc(current, labelsTest(target, :));
            end
        end
    end
    evaluation.EnsembleTestScore(target, :) = total / (iterations * folds);
    if testAvailable(target) && testCount > 0
        evaluation.EnsembleTestAUROC(target) = metric_auroc(evaluation.EnsembleTestScore(target, :), ...
            labelsTest(target, :));
    end
    total = zeros(1, sampleCount);
    complete = true;
    for iteration = 1:iterations
        current = double(outOfFold{target}(iteration, :));
        if all(isfinite(current))
            evaluation.OutOfFoldAUROC(iteration, target) = metric_auroc(current, labelsData(target, :));
        else
            complete = false;
        end
        total = total + current;
    end
    if complete
        evaluation.MeanOutOfFoldScore(target, :) = total / iterations;
    else
        evaluation.MeanOutOfFoldScore(target, :) = NaN;
    end
end
testValues = zeros(3, iterations * folds);
for target = 1:3
    testValues(target, :) = reshape(evaluation.TestAUROC(:, :, target).', 1, []);
end
evaluation.TestAUROCMean = rowStatistic(testValues, 'mean');
evaluation.TestAUROCStd = rowStatistic(testValues, 'std');
evaluation.TestAUROCMin = rowStatistic(testValues, 'min');
evaluation.TestAUROCMax = rowStatistic(testValues, 'max');
evaluation.TestAUROCOverall = rowStatistic(evaluation.TestAUROCMean, 'mean');
outOfFoldValues = evaluation.OutOfFoldAUROC.';
evaluation.OutOfFoldAUROCMean = rowStatistic(outOfFoldValues, 'mean');
evaluation.OutOfFoldAUROCStd = rowStatistic(outOfFoldValues, 'std');
evaluation.OutOfFoldAUROCMin = rowStatistic(outOfFoldValues, 'min');
evaluation.OutOfFoldAUROCMax = rowStatistic(outOfFoldValues, 'max');

end

function result = rowStatistic(values, kind)

result = nan(1, size(values, 1));
for row = 1:size(values, 1)
    entries = values(row, :);
    count = numel(entries);
    if count == 0 || any(isnan(entries))
        continue
    end
    total = 0;
    for index = 1:count
        total = total + entries(index);
    end
    average = total / count;
    switch kind
        case 'mean'
            result(row) = average;
        case 'std'
            if count > 1
                squares = 0;
                for index = 1:count
                    deviation = entries(index) - average;
                    squares = squares + deviation * deviation;
                end
                result(row) = sqrt(squares / (count - 1));
            end
        case 'min'
            result(row) = min(entries);
        case 'max'
            result(row) = max(entries);
    end
end

end
