function leo_cluster_tmas(bb,b)
rng(1);

sz=45; % this is for 20x, 448 pathces
sz=180; % this is for 40x, 224 patches

%d=dir('leo_feats/cmyc/*.h5');
%d=d(3:53);
d=dir('leo_feats/bcl2/*.h5');
d=d(5:end);
d=d(randperm(length(d)));
l=round(linspace(0,length(d),bb+1));
d=d(l(b)+1:l(b+1));
for i=2:4
    s=strsplit(d(i).name,'_'); s=str2num(s{1});
    %f=h5read(fullfile(d(i).folder,strcat(num2str(s),'.h5')),'/features');
    f=h5read(fullfile(d(i).folder,d(i).name),'/features');
    xs=h5read(fullfile(d(i).folder,d(i).name),'/xs');
    ys=h5read(fullfile(d(i).folder,d(i).name),'/ys');

    min_k=2;
    max_k=2048;
    while min_k<=max_k
        k=floor((min_k+max_k)/2);
        c=kmeans(cat(2,xs,ys),k);
        a=countlabels(c);
        if mean(a.Count)<sz
            max_k=k-1;
        elseif mean(a.Count)>sz
            min_k=k+1;
        else
            break;
        end
    end

    for v=1:k
        features=f(:,c==v);
        x=xs(c==v);
        y=ys(c==v);
        h5create(strcat('leo_feats_tmas/bcl2/40x/',num2str(s),'_',num2str(v),'.h5'),'/features',size(features),'Datatype','single');
        h5write(strcat('leo_feats_tmas/bcl2/40x/',num2str(s),'_',num2str(v),'.h5'),'/features',features);
        h5create(strcat('leo_feats_tmas/bcl2/40x/',num2str(s),'_',num2str(v),'.h5'),'/xs',size(x),'Datatype','single');
        h5write(strcat('leo_feats_tmas/bcl2/40x/',num2str(s),'_',num2str(v),'.h5'),'/xs',x);
        h5create(strcat('leo_feats_tmas/bcl2/40x/',num2str(s),'_',num2str(v),'.h5'),'/ys',size(y),'Datatype','single');
        h5write(strcat('leo_feats_tmas/bcl2/40x/',num2str(s),'_',num2str(v),'.h5'),'/ys',y);
    end
end

end