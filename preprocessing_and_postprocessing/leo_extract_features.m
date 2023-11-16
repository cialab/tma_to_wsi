function leo_extract_features(nnodes,node)
rng(2);

addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/include/openslide');
addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/lib');
openslide_load_library();

%d=cat(1,dir('bcl2/*.ndpi'),dir('cmyc/*.ndpi'));
d=dir('bcl2/*.ndpi');
d=d(randperm(length(d)));
d=d(14:18); %temp

ps=224;%448;
scale=1;%0.5;
net=resnet50;
wd='leo_feats';

l=round(linspace(0,length(d),nnodes+1));
for i=l(node)+1:l(node+1)
    
    fp=openslide_open(fullfile(d(i).folder,d(i).name));
    s=strsplit(d(i).name,'.');s=s{1};
    ss=strsplit(d(i).folder,'/'); ss=ss{end};
    if ~exist(strcat(ss,'_masks/',s,'.png'),'file')
        continue;
    end

    if exist(strcat(wd,'/',ss,'/',s,'.h5'),'file')
        continue;
    end

    mask=imread(strcat(ss,'_masks/',s,'.png'));
    [w,h]=openslide_get_level_dimensions(fp,0);

    rx=double(w)/size(mask,2);
    ry=double(h)/size(mask,1);
    ys=(ps/ry)/2:ps/ry:size(mask,1)-(ps/ry);
    xs=(ps/rx)/2:ps/rx:size(mask,2)-(ps/rx);
    [xs,ys]=meshgrid(xs,ys);
    xs=xs(:);
    ys=ys(:);
    keep=zeros(length(xs),1,'logical');
    for p=1:length(xs)
        if mask(round(ys(p)),round(xs(p)))
            keep(p)=1;
        end
    end
    xs=xs(keep);
    ys=ys(keep);

    xs=xs.*rx-ps/2+1;
    ys=ys.*ry-ps/2+1;
    ys(xs<0)=[];
    xs(xs<0)=[];
    xs(ys<0)=[];
    ys(ys<0)=[];
    ys(xs+ps-1>w)=[];
    xs(xs+ps-1>w)=[];
    xs(ys+ps-1>h)=[];
    ys(ys+ps-1>h)=[];
    patches=zeros(ps*scale,ps*scale,3,length(xs),'uint8');
    for p=1:length(xs)
        im=openslide_read_region(fp,xs(p),ys(p),ps,ps);
        im=im(:,:,2:4);
        im=imresize(im,scale);
        patches(:,:,:,p)=im;
    end

    if mod(size(patches,4),1024)~=0
        ll=[0:1024:size(patches,4) size(patches,4)];
    end

    feats=[];
    for j=1:length(ll)-1
        f=activations(net,patches(:,:,:,ll(j)+1:ll(j+1)),'activation_40_relu');
        f=squeeze(mean(f,[1 2]));
        feats=cat(2,feats,f);
    end
    h5create(strcat(wd,'/',ss,'/',s,'.h5'),'/features',size(feats),'Datatype','single');
    h5write(strcat(wd,'/',ss,'/',s,'.h5'),'/features',feats);
end

    
