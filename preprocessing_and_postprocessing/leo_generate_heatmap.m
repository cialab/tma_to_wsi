d=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/eval_results/EVAL_stanford_myc_by_patient_on_leo_wsi_as_tmas/fold_*.csv');
t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo312_tma/leo312_clinical.csv');
slides=table;
slides.slide_id=t.deid_id;
slides.label=t.cmyc_wsi_score_dj;

% For fold 1
%   #   gt  pr
%   107 45  43
%   117 65  14

% Each fold separately
for i=1:length(d)
    t=readtable(fullfile(d(i).folder,d(i).name));
    for j=1:size(slides,1)
        s=slides.slide_id(j);
        if s==60 % this has two slides
            continue;
        end
        tt=t(startsWith(t.slide_id,strcat(num2str(slides.slide_id(j)),'_')),:);
        
        fp=openslide_open(strcat('cmyc/',num2str(s,'%04.0f'),'_CMYC_01.ndpi'));
        [w0,h0]=openslide_get_level0_dimensions(fp);

        hm=zeros(round(h0/448),round(w0/448),'single');
        for k=1:size(tt,1)
            xs=h5read(strcat('leo_feats_tmas/cmyc/',tt.slide_id{k},'.h5'),'/xs');
            ys=h5read(strcat('leo_feats_tmas/cmyc/',tt.slide_id{k},'.h5'),'/ys');
            xs=round(xs./448);
            ys=round(ys./448);
            v=tt.Y_hat(k);
            for p=1:length(xs)
                hm(ys(p),xs(p))=v;
            end
        end
        hm(hm<0)=0;
        hm=hm-min(hm(:));
        hm=hm./max(hm(:));

        [w,h]=openslide_get_level_dimensions(fp,6);
        im=openslide_read_region(fp,0,0,w,h,'level',6);
        im=im(:,:,2:4);
        im=imresize(im,size(hm));
        
        subplot(1,2,1);
        imshow(im);
        subplot(1,2,2);
        imshow(hm);
        
        title('pick high point');
        [x,y]=ginput(1);
        
        % Insert shape
        hm=insertShape(hm,"rectangle",[x-3,y-3,7,7],'Color','red');
        im=insertShape(im,"rectangle",[x-3,y-3,7,7],'Color','red');

        x1=round((x-5)*448);
        y1=round((y-5)*448);
        w1=round(11*448);
        h1=round(11*448);
        imr=openslide_read_region(fp,x1,y1,w1,h1);
        imr=imr(:,:,2:4);
        imwrite(imr,strcat('leo_high_low_rois/',num2str(s),'_high_roi.png'));

        subplot(1,2,1);
        imshow(im);
        subplot(1,2,2);
        imshow(hm);

        title('pick low point');
        [x,y]=ginput(1);

        % Insert shape
        hm=insertShape(hm,"rectangle",[x-3,y-3,7,7],'Color','red');
        im=insertShape(im,"rectangle",[x-3,y-3,7,7],'Color','red');

        x1=round((x-5)*448);
        y1=round((y-5)*448);
        w1=round(11*448);
        h1=round(11*448);
        imr=openslide_read_region(fp,x1,y1,w1,h1);
        imr=imr(:,:,2:4);
        imwrite(imr,strcat('leo_high_low_rois/',num2str(s),'_low_roi.png'));

        % Write macros
        imwrite(im,strcat('leo_high_low_rois/',num2str(s),'_macro_cmyc.png'));
        imwrite(hm,strcat('leo_high_low_rois/',num2str(s),'_macro_hm.png'));
    end
end


