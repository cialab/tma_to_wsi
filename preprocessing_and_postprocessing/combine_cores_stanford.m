%d=dir('stanford_features/myc_40_40x/*.h5');
d=dir('stanford_features/bcl2_40_20x/*.h5');

n=cellfun(@(x) strsplit(x,'_'),{d.name},'UniformOutput',false);
n=vertcat(n{:});
n=unique(n(:,1));
for i=1:length(n)
    d=dir(strcat('stanford_features/bcl2_40_20x/',n{i},'*.h5'));
    feats=[];
    for j=1:length(d)
        f=h5read(fullfile(d(j).folder,d(j).name),'/features');
        l=h5read(fullfile(d(j).folder,d(j).name),'/label');
        feats=cat(2,feats,f);
    end
    h5create(strcat('stanford_features/bcl2_40_20x_by_patient/',n{i},'.h5'),'/features',size(feats),'Datatype','single');
    h5write(strcat('stanford_features/bcl2_40_20x_by_patient/',n{i},'.h5'),'/features',feats);
    h5create(strcat('stanford_features/bcl2_40_20x_by_patient/',n{i},'.h5'),'/label',size(l));
    h5write(strcat('stanford_features/bcl2_40_20x_by_patient/',n{i},'.h5'),'/label',l);
end
    

