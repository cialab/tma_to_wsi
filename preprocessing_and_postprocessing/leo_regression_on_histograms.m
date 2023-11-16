d=dir('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/eval_results/EVAL_stanford_bcl2_by_patient_on_leo_wsi_as_tmas/fold_*.csv');
t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo312_tma/leo312_clinical.csv');
slides=table;
slides.slide_id=t.deid_id;
slides.label=t.cmyc_wsi_score_dj;

% Each fold separately
corrs=[];
gts=[];
prs=[];
for i=1:length(d)
    t=readtable(fullfile(d(i).folder,d(i).name));
    hists=[];
    c=[];
    for j=1:size(slides,1)
        tt=t(startsWith(t.slide_id,strcat(num2str(slides.slide_id(j)),'_')),:);
        if size(tt,1)==0
            continue;
        end
        hc=histcounts(tt.Y_hat,0:5:100);
        hc=hc./max(hc);
        hists=cat(1,hists,hc);
        c=cat(1,c,slides.label(j));
    end

    % Do regression
    X=hists;
    Y=c;
    Yh=zeros(length(Y),1);
    parfor j=1:length(Y)
        XX=X;
        XX(j,:)=[];
        YY=Y;
        YY(j)=[];
        hyperopts = struct('AcquisitionFunctionName','expected-improvement-plus','MaxObjectiveEvaluations',100,'Verbose',0,'ShowPlots',false);
        [mdl,FitInfo,HyperparameterOptimizationResults] = fitrlinear(XX,YY,...
        'OptimizeHyperparameters','auto',...
        'HyperparameterOptimizationOptions',hyperopts);
        Yh(j)=predict(mdl,X(j,:));
    end
    corrs=cat(1,corrs,corr(Y,Yh,'type','Pearson'));
    gts=cat(1,gts,Y);
    prs=cat(1,prs,Yh);
end
fprintf('%0.4f +/- %0.4f\n',mean(corrs),std(corrs));