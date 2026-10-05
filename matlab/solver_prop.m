function solution = solver_prop(factor, rightHandSide, transposeSystem)

if nargin < 3
    transposeSystem = false;
end
if any(~isfinite(rightHandSide(:)))
    error('NeuroFANN:NonfinitePropagation', ...
        'The propagation right-hand side contains nonfinite values.');
end
if transposeSystem
    solution = factor' \ rightHandSide;
else
    solution = factor \ rightHandSide;
end
if any(~isfinite(solution(:)))
    error('NeuroFANN:NonfinitePropagation', ...
        'The propagation solve produced nonfinite values.');
end

end
