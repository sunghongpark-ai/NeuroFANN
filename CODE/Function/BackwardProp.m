function model = BackwardProp(model)

dLdBabt = (1/model.NumTrain)*model.ZTrain*(model.PabtTrain-model.YabtTrain)';
dLdBmta = (1/model.NumTrain)*model.ZTrain*(model.PmtaTrain-model.YmtaTrain)';
dLdBwmh = (1/model.NumTrain)*model.ZTrain*(model.PwmhTrain-model.YwmhTrain)';

dLdZabt = (1/model.NumTrain)*model.Babt*(model.PabtTrain-model.YabtTrain);
dLdZmta = (1/model.NumTrain)*model.Bmta*(model.PmtaTrain-model.YmtaTrain);
dLdZwmh = (1/model.NumTrain)*model.Bwmh*(model.PwmhTrain-model.YwmhTrain);
dLdZ = dLdZabt + dLdZmta + dLdZwmh;

dLdAclus = zeros(size(model.Aclus));
dLdH = zeros(size(model.HTrain));
for cluster = 1:model.NumCluster
    Eclus = find(model.IdxCluster==cluster);
    dLdAclus(Eclus) = (diag(model.Aprob(Eclus))-(model.Aprob(Eclus)*model.Aprob(Eclus)'))*(model.HTrain(Eclus,:)*dLdZ(cluster,:)');
    dLdH(Eclus,:) = model.Aprob(Eclus)*dLdZ(cluster,:);
end

dLdUprot = diag(dLdH*((model.Udiag+model.Lppi)*(eye(model.NumProtein)-((model.Udiag+model.Lppi)\model.Udiag))*model.XTrain)');

% Gradient
gBabt = dLdBabt + 2*model.RegCoeff*model.Babt;
gBmta = dLdBmta + 2*model.RegCoeff*model.Bmta;
gBwmh = dLdBwmh + 2*model.RegCoeff*model.Bwmh;
gAclus = dLdAclus + 2*model.RegCoeff*model.Aclus;
gUprot = dLdUprot + 2*model.RegCoeff*model.Uprot;
model.Gradient = [gUprot(:);gAclus(:);gBabt(:);gBmta(:);gBwmh(:)];