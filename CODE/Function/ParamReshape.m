function model = ParamReshape(model)

sizeUprot = model.SizeParam(1,:); start = 1                    ; finish = prod(sizeUprot)       ; model.Uprot = reshape(model.WeightParam(start:finish),sizeUprot);
sizeAclus = model.SizeParam(2,:); start = start+prod(sizeUprot); finish = finish+prod(sizeAclus); model.Aclus = reshape(model.WeightParam(start:finish),sizeAclus);
sizeBabt = model.SizeParam(3,:) ; start = start+prod(sizeAclus); finish = finish+prod(sizeBabt) ; model.Babt = reshape(model.WeightParam(start:finish),sizeBabt);
sizeBmta = model.SizeParam(4,:) ; start = start+prod(sizeBabt) ; finish = finish+prod(sizeBmta) ; model.Bmta = reshape(model.WeightParam(start:finish),sizeBmta);
sizeBwmh = model.SizeParam(5,:) ; start = start+prod(sizeBmta) ; finish = finish+prod(sizeBwmh) ; model.Bwmh = reshape(model.WeightParam(start:finish),sizeBwmh);

model.Udiag = diag(model.Uprot);
model.Aprob = zeros(model.NumProtein,1);
for cluster = 1:model.NumCluster
    Eclus = find(model.IdxCluster==cluster);
    model.Aprob(Eclus) = exp(model.Aclus(Eclus))./sum(exp(model.Aclus(Eclus)));
end