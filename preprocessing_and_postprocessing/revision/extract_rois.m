addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/include/openslide');
addpath('/isilon/datalake/cialab/scratch/cialab/tet/.usr/lib');
openslide_load_library();

roi_w=2*2048;
roi_h=2*1152;
d=dir('wsi_heatmaps/cmyc/fold_0/*.mat');

i=19; % 121
load(fullfile(d(i).folder,d(i).name));
fp=openslide_open(slide_path);
[w,h]=openslide_get_level_dimensions(fp,0);
w=double(w); h=double(h);
cm=jet(256);
hm2=ind2rgb(uint8(255.*hm2),cm);
imwrite(hm2,strcat('rois/121_attention_map.png'));

% get high attention
imshow(hm2);
[x,y]=getpts;
pts=cat(2,x,y);
for p=1:length(x)
    xx=round(x(p)*(w/size(hm2,2)));
    yy=round(y(p)*(h/size(hm2,1)));
    im=openslide_read_region(fp,xx-roi_w/2,yy-roi_h/2,roi_w,roi_h);
    im=im(:,:,2:4);
    imwrite(im,strcat('rois/121_high_attention_',num2str(p),'.png'));

    y1=round(y(p)-(roi_h/2)/(h/size(hm2,2)));
    x1=round(x(p)-(roi_w/2)/(w/size(hm2,1)));
    h1=round(roi_h/(h/size(hm2,1)));
    w1=round(roi_w/(w/size(hm2,2)));
    rois=zeros(size(hm2,1),size(hm2,2),3,'uint8');
    rois=insertShape(rois,'rectangle',[x1 y1 w1 h1],'Color','white');
    imwrite(rois,strcat('rois/121_high_attention_map_',num2str(p),'.png'),'Alpha',double(rois(:,:,1)==255));
end

% get low attention
imshow(hm2);
[x,y]=getpts;
pts=cat(2,x,y);
for p=1:length(x)
    xx=round(x(p)*(w/size(hm2,2)));
    yy=round(y(p)*(h/size(hm2,1)));
    im=openslide_read_region(fp,xx-roi_w/2,yy-roi_h/2,roi_w,roi_h);
    im=im(:,:,2:4);
    imwrite(im,strcat('rois/121_low_attention_',num2str(p),'.png'));

    y1=round(y(p)-(roi_h/2)/(h/size(hm2,2)));
    x1=round(x(p)-(roi_w/2)/(w/size(hm2,1)));
    h1=round(roi_h/(h/size(hm2,1)));
    w1=round(roi_w/(w/size(hm2,2)));
    rois=zeros(size(hm2,1),size(hm2,2),3,'uint8');
    rois=insertShape(rois,'rectangle',[x1 y1 w1 h1],'Color','white');
    imwrite(rois,strcat('rois/121_low_attention_map_',num2str(p),'.png'),'Alpha',double(rois(:,:,1)==255));
end




i=47; % 60
load(fullfile(d(i).folder,d(i).name));
fp=openslide_open(slide_path);
[w,h]=openslide_get_level_dimensions(fp,0);
w=double(w); h=double(h);
cm=jet(256);
hm2=ind2rgb(uint8(255.*hm2),cm);
imwrite(hm2,strcat('rois/60_attention_map.png'));

% get high attention
imshow(hm2);
[x,y]=getpts;
pts=cat(2,x,y);
for p=1:length(x)
    xx=round(x(p)*(w/size(hm2,2)));
    yy=round(y(p)*(h/size(hm2,1)));
    im=openslide_read_region(fp,xx-roi_w/2,yy-roi_h/2,roi_w,roi_h);
    im=im(:,:,2:4);
    imwrite(im,strcat('rois/60_high_attention_',num2str(p+16),'.png'));

    y1=round(y(p)-(roi_h/2)/(h/size(hm2,2)));
    x1=round(x(p)-(roi_w/2)/(w/size(hm2,1)));
    h1=round(roi_h/(h/size(hm2,1)));
    w1=round(roi_w/(w/size(hm2,2)));
    rois=zeros(size(hm2,1),size(hm2,2),3,'uint8');
    rois=insertShape(rois,'rectangle',[x1 y1 w1 h1],'Color','white');
    imwrite(rois,strcat('rois/60_high_attention_map_',num2str(p+16),'.png'),'Alpha',double(rois(:,:,1)==255));
end

