function model = ParamUpdate(model)

model.AdamParam.t = model.AdamParam.t + 1;
model.AdamParam.m = model.AdamParam.beta1*model.AdamParam.m + (1-model.AdamParam.beta1)*model.Gradient;
model.AdamParam.v = model.AdamParam.beta2*model.AdamParam.v + (1-model.AdamParam.beta2)*(model.Gradient.^2);
model.AdamParam.m_hat = model.AdamParam.m / (1 - model.AdamParam.beta1^model.AdamParam.t);
model.AdamParam.v_hat = model.AdamParam.v / (1 - model.AdamParam.beta2^model.AdamParam.t);
model.WeightParam = model.WeightParam - model.AdamParam.alpha * model.AdamParam.m_hat ./ (sqrt(model.AdamParam.v_hat) + model.AdamParam.epsilon);