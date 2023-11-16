function patch_stuff(bb,b)
rng(1);

addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/include/openslide');
addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/lib');
openslide_load_library();
d=cat(1,dir('bcl2/*.ndpi*'),dir('cmyc/*.ndpi*'));
r=randperm(length(d));
d=d(r);

ps=448;
stride=448;
scaling=0.5;

l=round(linspace(0,length(d),bb+1));
pp=parpool(8);
parfor i=l(b)+1:l(b+1)
    fp=openslide_open(fullfile(d(i).folder,d(i).name));
    [w0,h0]=openslide_get_level0_dimensions(fp); w0=double(w0); h0=double(h0);

    % Query points
    xq=1:stride:w0-ps;
    yq=1:stride:h0-ps;
    [xq,yq]=meshgrid(xq,yq);
    xq=xq(:); yq=yq(:);
    
    % Directory stuff
    s=strsplit(d(i).name,'.'); s=s{1};
    dd=strsplit(d(i).folder,'/'); dd=dd{end};
    if ~exist(strcat('patches/',dd,'/',s),'dir') 
        mkdir(strcat('patches/',dd,'/',s));
    end
    for p=1:length(xq)
        im=openslide_read_region(fp,xq(p),yq(p),ps,ps);
        im=imresize(im(:,:,2:4),scaling);
        if entropy(rgb2gray(im))>3
            imwrite(im,strcat('patches/',dd,'/',s,'/',num2str(xq(p)),'_',num2str(yq(p)),'.png'));
        end
    end
end
delete(pp);
end