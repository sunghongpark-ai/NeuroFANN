function model = ParamInit(model)

RandStream.setGlobalStream(RandStream('mt19937ar','Seed',model.IdxIter))

Uprot = zeros(model.NumProtein,1); sizeUprot = size(Uprot); model.Uprot = ones(sizeUprot);
Aclus = zeros(model.NumProtein,1); sizeAclus = size(Aclus); model.Aclus = zeros(sizeAclus);
Babt = zeros(model.NumCluster,1); sizeBabt = size(Babt); model.Babt = (2*rand(sizeBabt)-1)*sqrt(6/sum(sizeBabt));
Bmta = zeros(model.NumCluster,1); sizeBmta = size(Bmta); model.Bmta = (2*rand(sizeBmta)-1)*sqrt(6/sum(sizeBmta));
Bwmh = zeros(model.NumCluster,1); sizeBwmh = size(Bwmh); model.Bwmh = (2*rand(sizeBwmh)-1)*sqrt(6/sum(sizeBwmh));

model.SizeParam = [sizeUprot;sizeAclus;sizeBabt;sizeBmta;sizeBwmh];
model.NumParam = prod(sizeUprot)+prod(sizeAclus)+prod(sizeBabt)+prod(sizeBmta)+prod(sizeBwmh);
model.WeightParam = [model.Uprot(:);model.Aclus(:);model.Babt(:);model.Bmta(:);model.Bwmh(:)];
model.WeightEpoch = cell(model.MaxEpoch,1); model.LossTrain = zeros(model.MaxEpoch,1); model.LossValid = zeros(model.MaxEpoch,1);