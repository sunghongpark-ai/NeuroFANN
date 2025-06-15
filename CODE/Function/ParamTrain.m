function model = ParamTrain(model)

for IdxEpoch = 1:model.MaxEpoch
    model.IdxEpoch = IdxEpoch;
    model.WeightEpoch{model.IdxEpoch,1} = model.WeightParam;
    model = ParamReshape(model);
    model = ForwardProp(model);
    model = LossMeasure(model);
    model = BackwardProp(model);
    model = ParamUpdate(model);
end

model.BestEpoch = find(model.LossValid==min(model.LossValid),1);
model.WeightParam = model.WeightEpoch{model.BestEpoch};
model = ParamReshape(model);
model = ForwardProp(model);