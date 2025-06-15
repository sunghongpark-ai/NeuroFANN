function model = ForwardProp(model)

model.HTrain = (model.Udiag+model.Lppi)\model.Udiag*model.XTrain;
model.HValid = (model.Udiag+model.Lppi)\model.Udiag*model.XValid;
model.HTest = (model.Udiag+model.Lppi)\model.Udiag*model.XTest;

model.ZTrain = zeros(model.NumCluster,model.NumTrain);
model.ZValid = zeros(model.NumCluster,model.NumValid);
model.ZTest = zeros(model.NumCluster,model.NumTest);
for cluster = 1:model.NumCluster
    Eclus = find(model.IdxCluster==cluster);
    model.ZTrain(cluster,:) = model.Aprob(Eclus)'*model.HTrain(Eclus,:);
    model.ZValid(cluster,:) = model.Aprob(Eclus)'*model.HValid(Eclus,:);
    model.ZTest(cluster,:) = model.Aprob(Eclus)'*model.HTest(Eclus,:);
end

model.PabtTrain = 1./(1+exp((-1)*(model.Babt'*model.ZTrain)));
model.PmtaTrain = 1./(1+exp((-1)*(model.Bmta'*model.ZTrain)));
model.PwmhTrain = 1./(1+exp((-1)*(model.Bwmh'*model.ZTrain)));

model.PabtValid = 1./(1+exp((-1)*(model.Babt'*model.ZValid)));
model.PmtaValid = 1./(1+exp((-1)*(model.Bmta'*model.ZValid)));
model.PwmhValid = 1./(1+exp((-1)*(model.Bwmh'*model.ZValid)));

model.PabtTest = 1./(1+exp((-1)*(model.Babt'*model.ZTest)));
model.PmtaTest = 1./(1+exp((-1)*(model.Bmta'*model.ZTest)));
model.PwmhTest = 1./(1+exp((-1)*(model.Bwmh'*model.ZTest)));