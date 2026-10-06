function laplacian = ppi_laplacian(adjacency)

if ~(isnumeric(adjacency) || islogical(adjacency)) || ~isreal(adjacency) || ~ismatrix(adjacency) || ...
        isempty(adjacency) || size(adjacency, 1) ~= size(adjacency, 2)
    error('NeuroFANN:InvalidNetwork', 'The PPI weight matrix must be a nonempty real square matrix.');
end
adjacency = full(double(adjacency));
if any(~isfinite(adjacency(:))) || any(adjacency(:) < 0)
    error('NeuroFANN:InvalidNetwork', 'PPI weights must be finite and nonnegative.');
end
if ~isequal(adjacency, adjacency.')
    error('NeuroFANN:InvalidNetwork', 'The PPI weight matrix must be exactly symmetric.');
end
count = size(adjacency, 1);
degree = zeros(count, 1);
for column = 1:count
    degree = degree + adjacency(:, column);
end
if any(~isfinite(degree))
    error('NeuroFANN:InvalidNetwork', 'PPI weighted degrees exceed the floating-point range.');
end
scale = zeros(count, 1);
connected = degree > 0;
scale(connected) = 1 ./ sqrt(degree(connected));
laplacian = eye(count) - bsxfun(@times, bsxfun(@times, scale, adjacency), scale.');

end
