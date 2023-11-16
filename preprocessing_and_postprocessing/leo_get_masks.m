d=cat(1,dir('bcl2/*.ndpi'),dir('cmyc/*.ndpi'));

mask_ds=64;
ps=448;
pss=ps/mask_ds;

for i=i:length(d)
    fp=openslide_open(fullfile(d(i).folder,d(i).name));

    % Mask
    c=openslide_get_level_count(fp);
    for j=0:c-1
        ds=openslide_get_level_downsample(fp,j);
        if round(ds)==mask_ds
            break;
        end
    end
    [w,h]=openslide_get_level_dimensions(fp,j);
    im=openslide_read_region(fp,0,0,w,h,'level',j);
    im=im(:,:,2:4);
    mask=rgb2gray(im);
    mask=entropyfilt(mask,ones(pss,pss));
    mask=mask./max(mask(:));
    mask=mask>graythresh(mask(:));
    mask=imerode(mask,ones(pss,pss));

    % Remove areas by hand
    c=1;
    while c
        mask=bwareafilt(mask,[10000 Inf]); % remove small areas
        imshowpair(im,mask);
        h=drawfreehand('Closed',0);
        pts=h.Position;
        x=pts(:,1);
        y=pts(:,2);
        [x,idx]=unique(x,'stable');
        y=y(idx);
        [y,idx]=unique(y,'stable');
        x=x(idx);
        pts=cat(2,x,y);
        if size(pts,1)==1 % remove part of the mask
            [B,n]=bwlabel(mask);
            found=0;
            for j=1:n
                maskk=(B==j);
                if maskk(round(y),round(x))==1
                    mask(maskk)=0;
                    found=1;
                end
            end
            if ~found % click in background, stop loop
                c=0;
            end
        elseif size(pts,1)>1 % split region off
            maskk=zeros(size(mask),'logical');
            xq=floor(min(x)):0.25:ceil(max(x));
            yy=interp1(x,y,xq);
            xq(isnan(yy))=[];
            yy(isnan(yy))=[];
            for j=1:length(xq)
                maskk(round(yy(j)),round(xq(j)))=1;
            end
            maskk=imdilate(maskk,ones(3,3));
            mask(maskk)=0;
        end
    end
    s=strsplit(d(i).name,'.');s=s{1};
    ss=strsplit(d(i).folder,'/'); ss=ss{end};
    imwrite(mask,strcat(ss,'_masks/',s,'.png'));
end
