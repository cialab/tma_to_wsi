function extract_features_stanford(bb,b)
rng(1);

addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/include/openslide');
addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/lib');
openslide_load_library();

t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00147/stanford_cores.csv');
r=randperm(size(t,1));
t=t(r,:);
t2=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00147/stanford_donors.csv');
rd='/isilon/datalake/cialab/original/cialab/image_database/d00147/TMA/MYC/';
wd='stanford_features/myc_40_40x/';

ps=224;
stride=112;
scaling=1;

net=resnet50;

l=round(linspace(0,size(t,1),bb+1));
for i=l(b)+1:l(b+1)
    label=t2.MYC_(find(t.donor(i)==t2.donor));
    core_num=strsplit(t.core{i},' '); core_num=core_num{2};
    if isnan(label); continue; end
    if exist(strcat(wd,num2str(t.donor(i)),'_',core_num,'.h5'),'file')
        continue;
    end

    s=strsplit(t.slide{i},'_'); s=s{1};
    fp=openslide_open(strcat(rd,s,'.svs'));
    [w,h]=openslide_get_level0_dimensions(fp);

    xs=round(str2num(t.x{i}));
    ys=round(str2num(t.y{i}));
    if length(xs)>4
        % boundary
        im=openslide_read_region(fp,min(xs),min(ys),max(xs)-min(xs),max(ys)-min(ys));
        im=im(:,:,2:4);
    else
        % bounding box
        im=openslide_read_region(fp,xs(1),ys(1),min(xs(3),w)-xs(1),min(ys(3),h)-ys(1));
        im=im(:,:,2:4);
    end

    xs=1:stride:size(im,2)-ps;
    ys=1:stride:size(im,1)-ps;
    [xs,ys]=meshgrid(xs,ys);
    xs=xs(:);
    ys=ys(:);
    imgs=[];
    for j=1:length(xs)
        img=im(ys(j):ys(j)+ps-1,xs(j):xs(j)+ps-1,:);
        if entropy(img)>5
            imgs=cat(4,imgs,imresize(img,scaling));
        end
    end

    feats=activations(net,imgs,'activation_40_relu');
    feats=squeeze(mean(feats,[1 2]));

    h5create(strcat(wd,num2str(t.donor(i)),'_',core_num,'.h5'),'/features',size(feats),'Datatype','single');
    h5write(strcat(wd,num2str(t.donor(i)),'_',core_num,'.h5'),'/features',feats);
    h5create(strcat(wd,num2str(t.donor(i)),'_',core_num,'.h5'),'/label',size(label));
    h5write(strcat(wd,num2str(t.donor(i)),'_',core_num,'.h5'),'/label',label);
end


