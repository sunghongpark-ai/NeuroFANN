function value = metric_auroc(scores, labels)

if ~isnumeric(scores) || ~isreal(scores) || ~isvector(scores) || any(~isfinite(scores(:)))
    error('NeuroFANN:InvalidScores', 'Scores must be a finite real vector.');
end
if ~(isnumeric(labels) || islogical(labels)) || ~isreal(labels) || ~isvector(labels) || ...
        numel(labels) ~= numel(scores) || any(labels(:) ~= 0 & labels(:) ~= 1)
    error('NeuroFANN:InvalidTargets', 'Labels must be a binary vector matching the scores.');
end
scores = double(scores(:));
positive = double(labels(:)) == 1;
positiveCount = sum(positive);
negativeCount = numel(positive) - positiveCount;
if positiveCount == 0 || negativeCount == 0
    value = NaN;
    return
end
[sortedScores, order] = sort(scores);
count = numel(sortedScores);
starts = [true; diff(sortedScores) ~= 0];
firstPosition = find(starts);
lastPosition = [firstPosition(2:end) - 1; count];
groupRanks = (firstPosition + lastPosition) / 2;
ranks = zeros(count, 1);
ranks(order) = groupRanks(cumsum(starts));
value = (sum(ranks(positive)) - positiveCount * (positiveCount + 1) / 2) / ...
    (positiveCount * negativeCount);

end
