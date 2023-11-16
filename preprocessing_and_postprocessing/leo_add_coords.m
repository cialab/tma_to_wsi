rng(1);

addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/include/openslide');
addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/lib');
openslide_load_library();

%d=cat(1,dir('bcl2/*.ndpi'),dir('cmyc/*.ndpi'));
d=dir('bcl2/*.ndpi');

ps=224;%448;
scale=1;%0.5;
wd='leo_feats';

for i=1:length(d)
    fp=openslide_open(fullfile(d(i).folder,d(i).name));
    s=strsplit(d(i).name,'.');s=s{1};
    ss=strsplit(d(i).folder,'/'); ss=ss{end};
    if ~exist(strcat(ss,'_masks/',s,'.png'),'file')
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
    
    h5create(strcat(wd,'/',ss,'/',s,'.h5'),'/xs',size(xs),'Datatype','single');
    h5write(strcat(wd,'/',ss,'/',s,'.h5'),'/xs',xs);
    h5create(strcat(wd,'/',ss,'/',s,'.h5'),'/ys',size(ys),'Datatype','single');
    h5write(strcat(wd,'/',ss,'/',s,'.h5'),'/ys',ys);
end