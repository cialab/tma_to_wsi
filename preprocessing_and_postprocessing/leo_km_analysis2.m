d=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/eval_results/EVAL_stanford_myc_by_patient_on_leo_wsi_as_tmas/fold_*.csv');
t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo312_tma/leo312_clinical.csv');
slides=table;
slides.slide_id=t.deid_id;
slides.efs_stat=t.efs_stat;
slides.efs_months=t.t_efs_mth;
slides.label=t.cmyc_wsi_score_dj;

gtz=[];
prz=[];
efz_stat=[];
efz_months=[];
for i=1:length(d)
    t=readtable(fullfile(d(i).folder,d(i).name));
    gts=[];
    prs=[];
    efs_stat=[];
    efs_months=[];
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
        efs_stat=cat(1,efs_stat,slides.efs_stat(j));
        efs_months=cat(1,efs_months,slides.efs_months(j));
    end
    gtz=cat(1,gtz,gts);
    prz=cat(1,prz,prs);
    efz_stat=cat(1,efz_stat,efs_stat);
    efz_months=cat(1,efz_months,efs_months);
end

% figure;
% hold on;
% ecdf(efz_months(prz>=40),'Function','survivor','Censoring',efz_stat(prz>=40)==1);
% ecdf(efz_months(prz<40),'Function','survivor','Censoring',efz_stat(prz<40)==1);
% title('cmyc model')
% hold off
logrank(cat(2,efz_months(prz>=40),efz_stat(prz>=40)==1),cat(2,efz_months(prz<40),efz_stat(prz<40)==1));

ps=[];
ths=[];
for i=1:1:100
    p=logrank2(cat(2,efz_months(prz>=i),efz_stat(prz>=i)==1),cat(2,efz_months(prz<i),efz_stat(prz<i)==1));
    ths=[ths i];
    ps=[ps p];
end
plot(ths,ps);
hold on;
plot([0 100],[0.05 0.05],'r--')
title('CMYC logrank at multiple thresholds');
xlabel('threshold')
ylabel('p value');
legend('','p=0.05');
hold off;
saveas(gcf,'leo_plots/cmyc_logranks.png');




d=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/eval_results/EVAL_stanford_bcl2_by_patient_on_leo_wsi_as_tmas/fold_*.csv');
t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo312_tma/leo312_clinical.csv');
slides=table;
slides.slide_id=t.deid_id;
slides.label=t.cmyc_wsi_score_dj;
slides.efs_stat=t.efs_stat;
slides.efs_months=t.t_efs_mth;
slides.label=t.bcl2_wsi_score_dj;

gtz=[];
prz=[];
efz_stat=[];
efz_months=[];
for i=1:length(d)
    t=readtable(fullfile(d(i).folder,d(i).name));
    gts=[];
    prs=[];
    efs_stat=[];
    efs_months=[];
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
        efs_stat=cat(1,efs_stat,slides.efs_stat(j));
        efs_months=cat(1,efs_months,slides.efs_months(j));
    end
    gtz=cat(1,gtz,gts);
    prz=cat(1,prz,prs);
    efz_stat=cat(1,efz_stat,efs_stat);
    efz_months=cat(1,efz_months,efs_months);
end

% figure;
% hold on;
% ecdf(efz_months(prz>=70),'Function','survivor','Censoring',efz_stat(prz>=70)==1);
% ecdf(efz_months(prz<70),'Function','survivor','Censoring',efz_stat(prz<70)==1);
% title('bcl2 model')
% hold off
logrank(cat(2,efz_months(prz>=70),efz_stat(prz>=70)==1),cat(2,efz_months(prz<70),efz_stat(prz<70)==1));

ps=[];
ths=[];
for i=1:1:100
    p=logrank2(cat(2,efz_months(prz>=i),efz_stat(prz>=i)==1),cat(2,efz_months(prz<i),efz_stat(prz<i)==1));
    ths=[ths i];
    ps=[ps p];
end
plot(ths,ps);
hold on;
plot([0 100],[0.05 0.05],'r--')
title('BCL2 logrank at multiple thresholds');
xlabel('threshold')
ylabel('p value');
legend('','p=0.05');
hold off;
saveas(gcf,'leo_plots/bcl2_logranks.png');