function factor = factor_prop(operator)

if ~(isnumeric(operator) || islogical(operator)) || ~isreal(operator) || ~ismatrix(operator) || ...
        isempty(operator) || size(operator, 1) ~= size(operator, 2) || any(~isfinite(nonzeros(operator)))
    error('NeuroFANN:NonfinitePropagation', ...
        'The propagation matrix must be square with finite real values.');
end
operator = full(double(operator));
[lowerFactor, upperFactor, permutation] = lu(operator, 'vector');
if any(~isfinite(lowerFactor(:))) || any(~isfinite(upperFactor(:)))
    error('NeuroFANN:NonfinitePropagation', ...
        'The propagation factorization produced nonfinite values.');
end
reciprocalCondition = rcond(operator);
if ~isfinite(reciprocalCondition) || reciprocalCondition <= eps
    error('NeuroFANN:SingularPropagation', ...
        'diag(Uprot) + Lppi is singular or numerically singular (reciprocal condition %.3g).', ...
        reciprocalCondition);
end
factor = struct('Lower', lowerFactor, 'Upper', upperFactor, ...
    'Permutation', permutation(:), 'ReciprocalCondition', reciprocalCondition);

end
