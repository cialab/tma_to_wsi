rng(1);

t=readtable('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/dataset_csv/bcl2_score.csv');
slides=t.slide_id;

for i=1:size(t,1)
    te=slides(i);
    idxs=cat(2,1:i-1,i+1:size(t,1));
    vl_idx=randsample(idxs,round(size(t,1)*0.15));
    tr_idx=setdiff(idxs,vl_idx);
    tr=slides(tr_idx);
    vl=slides(vl_idx);
    
    tt=table;
    nums=strsplit(num2str(0:length(tr)-1),' ');
    tt.(" ")=string(nums)';
    tt.train=string(tr);
    tt.val=cat(1,string(vl),repmat([""],[length(tr)-length(vl) 1]));
    tt.test=cat(1,string(te),repmat([""],[length(tr)-length(te) 1]));
    writetable(tt,strcat('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/splits/bcl2_score/splits_',num2str(i-1),'.csv'));
end

t=readtable('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/dataset_csv/cmyc_score.csv');
slides=t.slide_id;

for i=1:size(t,1)
    te=slides(i);
    idxs=cat(2,1:i-1,i+1:size(t,1));
    vl_idx=randsample(idxs,round(size(t,1)*0.15));
    tr_idx=setdiff(idxs,vl_idx);
    tr=slides(tr_idx);
    vl=slides(vl_idx);
    
    tt=table;
    nums=strsplit(num2str(0:length(tr)-1),' ');
    tt.(" ")=string(nums)';
    tt.train=string(tr);
    tt.val=cat(1,string(vl),repmat([""],[length(tr)-length(vl) 1]));
    tt.test=cat(1,string(te),repmat([""],[length(tr)-length(te) 1]));
    writetable(tt,strcat('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/splits/cmyc_score/splits_',num2str(i-1),'.csv'));
end