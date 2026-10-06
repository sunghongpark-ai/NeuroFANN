function param = init_adamopt(num_var, alpha, beta1, beta2, epsilon)

if nargin < 2 || isempty(alpha)
    alpha = 1e-4;
end
if nargin < 3 || isempty(beta1)
    beta1 = 0.9;
end
if nargin < 4 || isempty(beta2)
    beta2 = 0.999;
end
if nargin < 5 || isempty(epsilon)
    epsilon = 1e-8;
end
if ~isnumeric(num_var) || ~isreal(num_var) || ~isscalar(num_var) || ~isfinite(num_var) || ...
        num_var < 1 || num_var ~= fix(num_var)
    error('NeuroFANN:InvalidParameter', 'num_var must be a positive integer scalar.');
end
alpha = optimizerScalar(alpha, 'alpha');
beta1 = optimizerScalar(beta1, 'beta1');
beta2 = optimizerScalar(beta2, 'beta2');
epsilon = optimizerScalar(epsilon, 'epsilon');
if alpha <= 0 || epsilon <= 0
    error('NeuroFANN:InvalidOptimizerState', 'Adam alpha and epsilon must be positive.');
end
if beta1 < 0 || beta1 >= 1 || beta2 < 0 || beta2 >= 1
    error('NeuroFANN:InvalidOptimizerState', 'Adam beta1 and beta2 must lie in [0,1).');
end

param.alpha = alpha;
param.beta1 = beta1;
param.beta2 = beta2;
param.epsilon = epsilon;
param.t = 0;
param.m = zeros(double(num_var), 1);
param.v = zeros(double(num_var), 1);

end

function value = optimizerScalar(value, name)

if ~isnumeric(value) || ~isreal(value) || ~isscalar(value) || ~isfinite(value)
    error('NeuroFANN:InvalidOptimizerState', '%s must be a finite real numeric scalar.', name);
end
value = double(value);

end
