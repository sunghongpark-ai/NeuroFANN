function init = ModelInit(model,fold)

init.IdxIter = model.IdxIter; init.IdxFold = fold; init.NumFold = model.Parameter.NumFold;

init.IdxTrain = find(model.Dataset.CVindex(model.IdxIter,:)~=fold); init.IdxValid = find(model.Dataset.CVindex(model.IdxIter,:)==fold);
init.NumTrain = length(init.IdxTrain); init.NumValid = length(init.IdxValid); init.NumTest = size(model.Dataset.XTest,2);

init.XTrain = model.Dataset.XData(:,init.IdxTrain)    ; init.XValid = model.Dataset.XData(:,init.IdxValid)    ; init.XTest = model.Dataset.XTest;
init.YabtTrain = model.Dataset.YabtData(init.IdxTrain); init.YabtValid = model.Dataset.YabtData(init.IdxValid); init.YabtTest = model.Dataset.YabtTest;
init.YmtaTrain = model.Dataset.YmtaData(init.IdxTrain); init.YmtaValid = model.Dataset.YmtaData(init.IdxValid); init.YmtaTest = model.Dataset.YmtaTest;
init.YwmhTrain = model.Dataset.YwmhData(init.IdxTrain); init.YwmhValid = model.Dataset.YwmhData(init.IdxValid); init.YwmhTest = model.Dataset.YwmhTest;

init.Wppi = model.Dataset.Wppi; init.Lppi = model.Dataset.Lppi;

init.IdxProtein = model.Dataset.IdxProtein; init.NumProtein = model.Dataset.NumProtein;
init.IdxCluster = model.Dataset.IdxCluster; init.NumCluster = model.Dataset.NumCluster;

init.MaxEpoch = model.Parameter.MaxEpoch;
init.LearnRate = model.Parameter.LearnRate;
init.RegCoeff = model.Parameter.RegCoeff;