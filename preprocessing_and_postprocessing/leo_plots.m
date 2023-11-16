d=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/eval_results/EVAL_stanford_myc_by_patient_on_leo_wsi_as_tmas/fold_*.csv');
t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo312_tma/leo312_clinical.csv');
slides=table;
slides.slide_id=t.deid_id;
slides.label=t.cmyc_wsi_score_dj;
gtz=[];
prz=[];
for i=1:length(d)
    t=readtable(fullfile(d(i).folder,d(i).name));
    gts=[];
    prs=[];
    for j=1:size(slides,1)
        tt=t(startsWith(t.slide_id,strcat(num2str(slides.slide_id(j)),'_')),:);
        if size(tt,1)==0
            continue;
        end
        gts=cat(1,gts,slides.label(j));
        pr=tt.Y_hat;
        pr(pr<0)=0;
        pr(pr>100)=100;
        prs=cat(1,prs,median(pr));

    end
    prz=cat(2,prz,prs);
    gtz=cat(2,gtz,gts);
end
plot(mean(gtz,2),mean(prz,2),'.','MarkerSize',10);
title('CMYC',strcat("Pearson correlation: ",num2str(corr(mean(gtz,2),mean(prz,2),'type','Pearson'))));
xlabel('pathologist score');
ylabel('model score');
fontsize(gca,10,'pixels')
saveas(gcf,strcat('leo_plots/cmyc_gt_vs_pr.png'));

d=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/eval_results/EVAL_stanford_bcl2_by_patient_on_leo_wsi_as_tmas/fold_*.csv');
t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo312_tma/leo312_clinical.csv');
slides=table;
slides.slide_id=t.deid_id;
slides.label=t.bcl2_wsi_score_dj;
gtz=[];
prz=[];
for i=1:length(d)
    t=readtable(fullfile(d(i).folder,d(i).name));
    gts=[];
    prs=[];
    for j=1:size(slides,1)
        tt=t(startsWith(t.slide_id,strcat(num2str(slides.slide_id(j)),'_')),:);
        if size(tt,1)==0
            continue;
        end
        gts=cat(1,gts,slides.label(j));
        pr=tt.Y_hat;
        pr(pr<0)=0;
        pr(pr>100)=100;
        prs=cat(1,prs,median(pr));
    end
    prz=cat(2,prz,prs);
    gtz=cat(2,gtz,gts);
end
plot(mean(gtz,2),mean(prz,2),'.');
title('BCL2',strcat("Pearson correlation: ",num2str(corr(mean(gtz,2),mean(prz,2),'type','Pearson'))));
xlabel('pathologist score');
ylabel('model score');
saveas(gcf,strcat('leo_plots/bcl2_gt_vs_pr.png'));
