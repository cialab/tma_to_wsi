%d=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/eval_results/EVAL_stanford_myc_by_patient_on_leo_wsi_as_tmas/fold_*.csv');
d=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/eval_results/EVAL_stanford_bcl2_by_patient_on_leo_wsi_as_tmas/fold_*.csv');
t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo312_tma/leo312_clinical.csv');
slides=table;
slides.slide_id=t.deid_id;
%slides.label=t.cmyc_wsi_score_dj;
slides.label=t.bcl2_wsi_score_dj;

for i=1:length(d)
    t=readtable(fullfile(d(i).folder,d(i).name));
    gts=[];
    prs=[];
    for j=1:size(slides,1)
        tt=t(startsWith(t.slide_id,strcat(num2str(slides.slide_id(j)),'_')),:);
        if size(tt,1)==0
            continue;
        end
        histogram(tt.Y_hat,'BinEdges',0:5:100);
        title({strcat("pathologist score: ",num2str(slides.label(j))),...
               strcat("mini-bag prediction median: ",num2str(median(tt.Y_hat)))});
        xlabel('mini-bag BCL2 predictions')
        ylabel('# of mini-bags')
        saveas(gcf,strcat('leo_wsi_histograms/bcl2/',num2str(i),'/',num2str(slides.slide_id(j)),'.png'));
    end
end
