%d=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/eval_results/EVAL_stanford_myc_by_patient_on_leo_wsi_as_tmas/fold_*.csv');
d=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/eval_results/EVAL_stanford_bcl2_by_patient_on_leo_wsi_as_tmas/fold_*.csv');
t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo312_tma/leo312_clinical.csv');
slides=table;
slides.slide_id=t.deid_id;
%slides.label=t.cmyc_wsi_score_dj;
slides.label=t.bcl2_wsi_score_dj;

hists=[];
c=[];
for i=1:length(d)
    t=readtable(fullfile(d(i).folder,d(i).name));
    gts=[];
    prs=[];
    for j=1:size(slides,1)
        tt=t(startsWith(t.slide_id,strcat(num2str(slides.slide_id(j)),'_')),:);
        if size(tt,1)==0
            continue;
        end
        hists=cat(1,hists,histcounts(tt.Y_hat,0:5:100));
        c=cat(1,c,slides.label(j));
    end
end

D=pdist(hists);
D=squareform(D);
y2=mdscale(D,2);
y3=mdscale(D,3);

cm=jet();
c=c./100;
c=c.*255;

figure;
hold on;
for i=1:length(c)
    idx=round(c(i));
    plot(y2(i,1),y2(i,2),'*','Color',cm(idx+1,:));
end
xlabel('mds\_1')
ylabel('mds\_2')
hold off;

figure;
hold on;
for i=1:length(c)
    idx=round(c(i));
    plot3(y3(i,1),y3(i,2),y3(i,3),'*','Color',cm(idx+1,:));
end
xlabel('mds\_1')
ylabel('mds\_2')
zlabel('mds\_3')
hold off;
