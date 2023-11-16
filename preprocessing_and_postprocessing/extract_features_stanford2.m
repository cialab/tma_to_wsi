function extract_features_stanford2(bb,b)
rng(1);

addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/include/openslide');
addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/lib');
openslide_load_library();

t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00147/stanford_cores_updated.csv');
t=t(contains(t.slide,'BCL2'),:);
r=randperm(size(t,1));
t=t(r,:);
t2=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00147/stanford_donors.csv');
rd='/isilon/datalake/cialab/original/cialab/image_database/d00147/TMA/BCL2/';
wd='stanford_features/bcl2_40_20x/';

ps=448;
stride=112;
scaling=0.5;

net=resnet50;

l=round(linspace(0,size(t,1),bb+1));
for i=l(b)+1:l(b+1)
    label=t2.BCL2_(find(t.donor(i)==t2.donor));
    if isnan(label); continue; end

    c=t.core_doNotUseForCoreCorrespondenceBetweenCMYC_BCL2Slides_{i};
    c=strsplit(c,' '); c=c{2};
    if ~exist(strcat(wd,num2str(t.donor(i))),'dir'); mkdir(strcat(wd,num2str(t.donor(i)))); end

    s=strsplit(t.slide{i},'_'); s=s{1};
    fp=openslide_open(strcat(rd,s,'.svs'));
    [w,h]=openslide_get_level0_dimensions(fp);

    x=round(str2num(t.x{i})); y=round(str2num(t.y{i}));
    xs=min(x); xe=max(x);
    ys=min(y); ye=max(y);
    
    im=openslide_read_region(fp,xs,ys,xe-xs,ye-ys);
    im=im(:,:,2:4);

    xs=1:stride:size(im,2)-ps;
    ys=1:stride:size(im,1)-ps;
    [xs,ys]=meshgrid(xs,ys);
    xs=xs(:);
    ys=ys(:);
    imgs=[];
    for j=1:length(xs)
        img=im(ys(j):ys(j)+ps-1,xs(j):xs(j)+ps-1,:);
        if entropy(img)>5
            img=imresize(img,scaling);
            imgs=cat(4,imgs,img);
        end
    end

    feats=activations(net,imgs,'activation_40_relu');
    feats=squeeze(mean(feats,[1 2]));

    h5create(strcat(wd,num2str(t.donor(i)),'_',c,'.h5'),'/features',size(feats),'Datatype','single');
    h5write(strcat(wd,num2str(t.donor(i)),'_',c,'.h5'),'/features',feats);
    h5create(strcat(wd,num2str(t.donor(i)),'_',c,'.h5'),'/label',size(label));
    h5write(strcat(wd,num2str(t.donor(i)),'_',c,'.h5'),'/label',label);
end