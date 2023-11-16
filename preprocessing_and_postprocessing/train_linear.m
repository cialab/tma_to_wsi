function train_linear(f)
pp=parpool(16);

rd='/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/results/bcl2/feats/';
tr=readtable(strcat(rd,'simclr/nsclc_5folds/',num2str(f),'/training.csv'),'Delimiter',',');
vl=readtable(strcat(rd,'simclr/nsclc_5folds/',num2str(f),'/validation.csv'),'Delimiter',',');
te=readtable(strcat(rd,'simclr/nsclc_5folds/',num2str(f),'/testing.csv'),'Delimiter',',');
tr_slides=string(unique(tr.slide_id));
vl_slides=string(unique(vl.slide_id));
te_slides=string(unique(te.slide_id));
luad=string({dir('/isilon/datalake/cialab/scratch/cialab/tet/MATLAB/thesis/3/nsclc/patches/luad/TCGA*').name});
lusc=string({dir('/isilon/datalake/cialab/scratch/cialab/tet/MATLAB/thesis/3/nsclc/patches/lusc/TCGA*').name});

trX=cell(length(tr_slides),1);
trY=cell(length(tr_slides),1);
tridx=zeros(length(tr_slides),1);
parfor i=1:length(trX)
    trX{i}=h5read(strcat(rd,'datasets/nsclc_5folds/',num2str(f),'/feats/all/',tr_slides(i),'.h5'),'/features')';
    tridx(i)=size(trX{i},1);
    if ismember(tr_slides(i),luad)
        label="luad";
    elseif ismember(tr_slides(i),lusc)
        label="lusc";
    else
        fprint('label error\n');
        exit(0);
    end
    trY{i}=repmat([label],size(trX{i},1),1);
end
trX=cat(1,trX{:});
trY=cat(1,trY{:});
trY=categorical(trY);

vlX=cell(length(vl_slides),1);
vlY=cell(length(vl_slides),1);
vlidx=zeros(length(vl_slides),1);
parfor i=1:length(vlX)
    vlX{i}=h5read(strcat(rd,'datasets/nsclc_5folds/',num2str(f),'/feats/all/',vl_slides(i),'.h5'),'/features')';
    vlidx(i)=size(vlX{i},1);
    if ismember(tr_slides(i),luad)
        label="luad";
    elseif ismember(tr_slides(i),lusc)
        label="lusc";
    else
        fprint('label error\n');
        exit(0);
    end
    vlY{i}=repmat([label],size(vlX{i},1),1);
end
vlX=cat(1,vlX{:});
vlY=cat(1,vlY{:});
vlY=categorical(vlY);

delete(pp);

classWeights=countlabels(trY);
classes=classWeights.Label;
classWeights=1./(classWeights.Percent./100);
layers=[
    featureInputLayer(2048,'Name','input');
    fullyConnectedLayer(2048,'Name','fc1');
    reluLayer('Name','relu1');
    fullyConnectedLayer(2048,'Name','fc2');
    reluLayer('Name','relu2');
    fullyConnectedLayer(2,'Name','fc3');
    softmaxLayer('Name','softmax');
    classificationLayer('Name','classify','ClassWeights',classWeights,'Classes',classes)];

miniBatchSize=min(2^18,size(trX,1));
options = trainingOptions('sgdm', ...
    'MaxEpochs',500,...
    'VerboseFrequency',round(size(trX,1)/miniBatchSize),...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Verbose',true,...
    'ValidationData',{vlX,vlY},...
    'ValidationFrequency',round(size(trX,1)/miniBatchSize),...
    'OutputNetwork','best-validation-loss');
net = trainNetwork(trX,trY,layers,options);

clearvars trX vlX;

pr=predict(net,trX);
idx=0;
for i=1:length(tridx)
    [~,idy]=sort(max(pr(idx+1:idx+tridx(i),:),[],2),'descend');
    features=trX(idx+1:idx+tridx(i),:);
    features=features(idy,:);
    h5create(strcat('sorted_features/',num2str(f),'/',tr_slides(i),'.h5'),'/features',size(features),'Datatype','single');
    h5write(strcat('sorted_features/',num2str(f),'/',tr_slides(i),'.h5'),'/features',features);
    idx=idx+tridx(i);
end

pr=predict(net,vlX);
idx=0;
for i=1:length(vlidx)
    [~,idy]=sort(max(pr(idx+1:idx+vlidx(i),:),[],2),'descend');
    features=vlX(idx+1:idx+vlidx(i),:);
    features=features(idy,:);
    h5create(strcat('sorted_features/',num2str(f),'/',vl_slides(i),'.h5'),'/features',size(features),'Datatype','single');
    h5write(strcat('sorted_features/',num2str(f),'/',vl_slides(i),'.h5'),'/features',features);
    idx=idx+vlidx(i);
end

for i=1:length(te_slides)
    features=h5read(strcat(rd,'datasets/nsclc_5folds/',num2str(f),'/feats/all/',te_slides(i),'.h5'),'/features')';
    pr=predict(net,features);
    [~,idy]=sort(max(pr,[],2),'descend');
    features=features(idy,:);
    h5create(strcat('sorted_features/',num2str(f),'/',te_slides(i),'.h5'),'/features',size(features),'Datatype','single');
    h5write(strcat('sorted_features/',num2str(f),'/',te_slides(i),'.h5'),'/features',features);
end

end