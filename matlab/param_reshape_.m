function model = param_reshape_(model)

expectedSizes = [model.NumProtein, 1; model.NumProtein, 1; ...
    model.NumCluster, 1; model.NumCluster, 1; model.NumCluster, 1];
if ~isequal(model.SizeParam, expectedSizes) || ...
        ~isvector(model.WeightParam) || ...
        numel(model.WeightParam) ~= sum(prod(expectedSizes, 2))
    error('NeuroFANN:InvalidParameterSize', ...
        'The parameter vector and parameter shapes do not match the model.');
end
if ~isnumeric(model.WeightParam) || ~isreal(model.WeightParam) || ...
        ~isfloat(model.WeightParam) || any(~isfinite(model.WeightParam(:)))
    error('NeuroFANN:InvalidParameters', ...
        'WeightParam must contain finite real floating-point values.');
end

model.WeightParam = model.WeightParam(:);
names = {'Uprot', 'Aclus', 'Babt', 'Bmta', 'Bwmh'};
offset = 0;
for index = 1:numel(names)
    count = prod(expectedSizes(index, :));
    model.(names{index}) = reshape(model.WeightParam(offset + (1:count)), ...
        expectedSizes(index, :));
    offset = offset + count;
end

model.Udiag = spdiags(model.Uprot, 0, model.NumProtein, model.NumProtein);
clusterIndex = model.IdxCluster(:);
if ~isfield(model, 'ClusterIndexSnapshot') || ...
        ~isequal(model.ClusterIndexSnapshot, clusterIndex) || ...
        ~isfield(model, 'ClusterMembers') || ...
        numel(model.ClusterMembers) ~= model.NumCluster || ...
        ~isfield(model, 'ClusterAssignment') || ...
        ~isequal(size(model.ClusterAssignment), [model.NumCluster, model.NumProtein])
    if ~isnumeric(clusterIndex) || ~isreal(clusterIndex) || ...
            numel(clusterIndex) ~= model.NumProtein || any(~isfinite(clusterIndex)) || ...
            any(clusterIndex ~= fix(clusterIndex)) || ...
            any(clusterIndex < 1 | clusterIndex > model.NumCluster)
        error('NeuroFANN:InvalidClusters', ...
            'IdxCluster must assign one integer cluster index to every protein.');
    end
    model.ClusterMembers = cell(model.NumCluster, 1);
    for cluster = 1:model.NumCluster
        model.ClusterMembers{cluster} = find(clusterIndex == cluster);
    end
    model.ClusterAssignment = sparse(clusterIndex, (1:model.NumProtein).', ...
        1, model.NumCluster, model.NumProtein);
    model.ClusterIndexSnapshot = clusterIndex;
end
model.Aprob = zeros(size(model.Aclus), 'like', model.Aclus);
for cluster = 1:model.NumCluster
    members = model.ClusterMembers{cluster};
    if isempty(members)
        error('NeuroFANN:EmptyCluster', 'Every cluster must contain a protein.');
    end
    logits = model.Aclus(members);
    weights = exp(logits - max(logits));
    model.Aprob(members) = weights ./ sum(weights);
end
if isfield(model, 'PropagationFactor')
    model = rmfield(model, 'PropagationFactor');
end

end
