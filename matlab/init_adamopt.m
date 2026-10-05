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

validateattributes(num_var, {'numeric'}, {'scalar', 'real', 'finite', 'integer', 'positive'}, mfilename, 'num_var');
validateattributes(alpha, {'numeric'}, {'scalar', 'real', 'finite', 'positive'}, mfilename, 'alpha');
validateattributes(beta1, {'numeric'}, {'scalar', 'real', 'finite', '>=', 0, '<', 1}, mfilename, 'beta1');
validateattributes(beta2, {'numeric'}, {'scalar', 'real', 'finite', '>=', 0, '<', 1}, mfilename, 'beta2');
validateattributes(epsilon, {'numeric'}, {'scalar', 'real', 'finite', 'positive'}, mfilename, 'epsilon');

param.alpha = double(alpha);
param.beta1 = double(beta1);
param.beta2 = double(beta2);
param.epsilon = double(epsilon);
param.t = 0;
param.m = zeros(num_var, 1);
param.v = zeros(num_var, 1);
end

