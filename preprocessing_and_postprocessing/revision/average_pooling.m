function average_pooling(splits_dir,feats_dir)
d=dir(fullfile(feats_dir,'*.h5'));
sids=zeros(length(d),1);
X=cell(length(d),1);
Y=cell(length(d),1);
for i=1:length(d)
    X{i}=mean(h5read(fullfile(d(i).folder,d(i).name),'/features'),2);
    Y{i}=h5read(fullfile(d(i).folder,d(i).name),'/label');
    s=strsplit(d(i).name,'.');
    sids(i)=str2num(s{1});
end
X=cat(2,X{:})';
Y=cat(1,Y{:});

layers=[featureInputLayer(size(X,2))
        fullyConnectedLayer(size(X,2)/4)
        reluLayer()
        dropoutLayer(0.25)
        fullyConnectedLayer(1)
        regressionLayer()];

for s=1:10
    splits=readtable(fullfile('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/splits',splits_dir,strcat('splits_',num2str(s-1),'.csv')));
    [~,ia,~]=intersect(sids,splits.train);
    trX=X(ia,:);
    trY=Y(ia);
    [~,ia,~]=intersect(sids,splits.val);
    vlX=X(ia,:);
    vlY=Y(ia);
    [~,ia,~]=intersect(sids,splits.test);
    teX=X(ia,:);
    teY=Y(ia);

    miniBatchSize=min(2^18,size(trX,1));
    options = trainingOptions('adam', ...
        'MaxEpochs',200,...
        'VerboseFrequency',round(size(trX,1)/miniBatchSize),...
        'MiniBatchSize',miniBatchSize, ...
        'Shuffle','every-epoch', ...
        'Verbose',true,...
        'ValidationData',{vlX,vlY},...
        'ValidationFrequency',round(size(trX,1)/miniBatchSize),...
        'OutputNetwork','best-validation-loss',...
        'ValidationPatience',20);
    net = trainNetwork(trX,trY,layers,options);

    teYp=predict(net,teX);

    wd1=strsplit(splits_dir,{'_by_patient','/'});
    wd2=strsplit(feats_dir,{'_by_patient','/'});
    wd=fullfile('results',strcat('stanford_',wd2{3},wd1{2}));
    if ~exist(wd,'dir')
        mkdir(wd);
    end

    t=table(teY,teYp);
    writetable(t,fullfile(wd,strcat('splits_',num2str(s-1),'.csv')));
end