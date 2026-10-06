function model = param_update(model)

if ~isfloat(model.WeightParam) || ~isreal(model.WeightParam) || ~iscolumn(model.WeightParam) || ...
        isempty(model.WeightParam) || any(~isfinite(model.WeightParam))
    error('NeuroFANN:InvalidWeights', 'WeightParam must be a nonempty finite real floating-point column vector.');
end
if ~isfloat(model.Gradient) || ~isreal(model.Gradient) || ...
        ~isequal(size(model.Gradient), size(model.WeightParam)) || any(~isfinite(model.Gradient(:)))
    error('NeuroFANN:InvalidGradient', ...
        'Gradient must be a finite real floating-point column vector matching WeightParam.');
end
if ~isfield(model, 'AdamParam') || ~isstruct(model.AdamParam) || ~isscalar(model.AdamParam)
    error('NeuroFANN:InvalidOptimizerState', 'AdamParam must be a scalar structure created by init_adamopt.');
end
param = model.AdamParam;
required = {'alpha', 'beta1', 'beta2', 'epsilon', 't', 'm', 'v'};
if ~all(isfield(param, required))
    error('NeuroFANN:InvalidOptimizerState', ...
        'AdamParam is missing required optimizer fields. Initialize it with init_adamopt.');
end
if ~validScalar(param.alpha) || param.alpha <= 0 || ...
        ~validScalar(param.epsilon) || param.epsilon <= 0
    error('NeuroFANN:InvalidOptimizerState', 'Adam alpha and epsilon must be finite positive real numeric scalars.');
end
if ~validScalar(param.beta1) || param.beta1 < 0 || param.beta1 >= 1 || ...
        ~validScalar(param.beta2) || param.beta2 < 0 || param.beta2 >= 1
    error('NeuroFANN:InvalidOptimizerState', 'Adam beta1 and beta2 must be finite real numeric scalars in [0,1).');
end
if ~validScalar(param.t) || param.t < 0 || param.t >= flintmax || param.t ~= fix(param.t)
    error('NeuroFANN:InvalidOptimizerState', 'Adam t must be a nonnegative integer less than flintmax.');
end
if ~isfloat(param.m) || ~isreal(param.m) || ~isequal(size(param.m), size(model.WeightParam)) || ...
        any(~isfinite(param.m(:))) || ~isfloat(param.v) || ~isreal(param.v) || ...
        ~isequal(size(param.v), size(model.WeightParam)) || any(~isfinite(param.v(:))) || any(param.v(:) < 0)
    error('NeuroFANN:InvalidOptimizerState', ...
        'Adam moments must be finite real floating-point column vectors matching WeightParam, with nonnegative v.');
end

param.alpha = double(param.alpha);
param.beta1 = double(param.beta1);
param.beta2 = double(param.beta2);
param.epsilon = double(param.epsilon);
param.t = double(param.t) + 1;
gradient = double(model.Gradient);
param.m = param.beta1 * double(param.m) + (1 - param.beta1) * gradient;
param.v = param.beta2 * double(param.v) + (1 - param.beta2) * (gradient .^ 2);
param.m_hat = param.m / (1 - param.beta1 ^ param.t);
param.v_hat = param.v / (1 - param.beta2 ^ param.t);
weights = double(model.WeightParam) - param.alpha * param.m_hat ./ (sqrt(param.v_hat) + param.epsilon);

if any(~isfinite(param.m)) || any(~isfinite(param.v)) || any(~isfinite(param.m_hat)) || ...
        any(~isfinite(param.v_hat)) || any(~isfinite(weights))
    error('NeuroFANN:NonfiniteOptimizerUpdate', ...
        'Adam produced nonfinite state or weights. Check the gradient magnitude, learning rate, and input scaling.');
end
model.AdamParam = param;
model.WeightParam = weights;

end

function valid = validScalar(value)

valid = isnumeric(value) && isreal(value) && isscalar(value) && isfinite(value);

end
