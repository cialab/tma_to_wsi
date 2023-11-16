addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/include/openslide');
addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/lib');
openslide_load_library();
fp=openslide_open('cmyc/0101_CMYC_01.ndpi');
[w0,h0]=openslide_get_level_dimensions(fp,0);
[w,h]=openslide_get_level_dimensions(fp,5);
im=openslide_read_region(fp,0,0,w,h,'level',5);
im=im(:,:,2:4);
imwrite(im,'clusters_myc.png');

% rng(1);
% d=dir('leo_feats_tmas/cmyc/101*.h5');
% d=d(randperm(length(d)));
% img=zeros(round(h0/448),round(w0/448),3,'double');
% %cm=turbo(length(d));
% cm=jet(length(d));
% cm=cm(randperm(length(d)),:);
% for i=1:length(d)
%     xs=h5read(fullfile(d(i).folder,d(i).name),'/xs');
%     ys=h5read(fullfile(d(i).folder,d(i).name),'/ys');
%     xs=round(xs./448);
%     ys=round(ys./448);
%     for j=1:length(xs)
%         img(ys(j),xs(j),1)=cm(i,1);
%         img(ys(j),xs(j),2)=cm(i,2);
%         img(ys(j),xs(j),3)=cm(i,3);
%     end
% end
% imshow(img);

rng(1);
num_colors=20;
d=dir('leo_feats_tmas/cmyc/101*.h5');
d=d(randperm(length(d)));
img=zeros(round(h0/448),round(w0/448),'uint64');
imga=zeros(round(h0/448),round(w0/448),'uint64');
cm=jet(num_colors);
vals=1:num_colors;
v=0;
for i=1:length(d)
    xs=h5read(fullfile(d(i).folder,d(i).name),'/xs');
    ys=h5read(fullfile(d(i).folder,d(i).name),'/ys');
    xs=round(xs./448);
    ys=round(ys./448);
    imgt=zeros(round(h0/448),round(w0/448),'logical');
    for j=1:length(xs)
        imgt(ys(j),xs(j))=1;
    end
    
    [~,n]=bwlabel(img>0);
    [~,nt]=bwlabel((img>0)|(imgt>0));
    if n==nt % touching
        imgt2=imdilate(imgt,strel('disk',1,0));
        vals_used=[];
        for j=1:i-1
            imgj=(imga==j);
            if sum(sum(imgj&imgt2))>0
                val=img(imgj);
                val=unique(val);
                vals_used=cat(1,vals_used,val);
            end
        end
        s=setdiff(vals,vals_used);
        s=randsample(s,1);
        img(imgt)=s;
    else % not touching
        img(imgt)=vals(mod(v,num_colors)+1);
        v=v+1;
    end

    imga(imgt)=i;

%     img1=zeros(round(h0/448),round(w0/448),'double');
%     img2=zeros(round(h0/448),round(w0/448),'double');
%     img3=zeros(round(h0/448),round(w0/448),'double');
%     for j=1:num_colors
%         img1(img==j)=cm(j,1);
%         img2(img==j)=cm(j,2);
%         img3(img==j)=cm(j,3);
%     end
%     imgc=cat(3,img1,img2,img3);
%     imshow(imgc);
%     pause;
end
img1=ones(round(h0/448),round(w0/448),'double');
img2=ones(round(h0/448),round(w0/448),'double');
img3=ones(round(h0/448),round(w0/448),'double');
for j=1:num_colors
    img1(img==j)=cm(j,1);
    img2(img==j)=cm(j,2);
    img3(img==j)=cm(j,3);
end
imgc=cat(3,img1,img2,img3);
%imshow(imgc);
imgc=imresize(imgc,[size(im,1) size(im,2)],'nearest');
imwrite(imgc,'clusters.png');
