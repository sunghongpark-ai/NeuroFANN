function solution = solver_prop(factor, rightHandSide, transposeSystem)

if nargin < 3
    transposeSystem = false;
end
if ~isstruct(factor) || ~isscalar(factor) || ~all(isfield(factor, {'Lower', 'Upper', 'Permutation'}))
    error('NeuroFANN:InvalidPropagationFactor', 'Create the propagation factor with factor_prop.');
end
count = size(factor.Upper, 1);
if ~(isnumeric(rightHandSide) || islogical(rightHandSide)) || ~isreal(rightHandSide) || ...
        ~ismatrix(rightHandSide) || size(rightHandSide, 1) ~= count || ...
        any(~isfinite(nonzeros(rightHandSide)))
    error('NeuroFANN:NonfinitePropagation', ...
        'The propagation right-hand side must have one finite real row per protein.');
end
rightHandSide = full(double(rightHandSide));
if transposeSystem
    intermediate = factor.Upper.' \ rightHandSide;
    solution = zeros(size(rightHandSide));
    solution(factor.Permutation, :) = factor.Lower.' \ intermediate;
else
    solution = factor.Upper \ (factor.Lower \ rightHandSide(factor.Permutation, :));
end
if any(~isfinite(solution(:)))
    error('NeuroFANN:NonfinitePropagation', ...
        'The propagation solve produced nonfinite values.');
end

end
