d=dir('leo_feats/bcl2/*.h5');
cellfun(@(x) strsplit(x,'.'),{d.name},'UniformOutput',false);
vertcat(ans{:});
ans(:,1);
slides=cellfun(@(x) str2num(x),ans);

for i=1:length(slides)
    te=slides(i);
    idxs=cat(2,1:i-1,i+1:length(slides));
    vl_idx=randsample(idxs,round(length(slides)*0.15));
    tr_idx=setdiff(idxs,vl_idx);
    tr=slides(tr_idx);
    vl=slides(vl_idx);
    
    tt=table;
    nums=strsplit(num2str(0:length(tr)-1),' ');
    tt.(" ")=string(nums)';
    tt.train=string(tr);
    tt.val=cat(1,string(vl),repmat([""],[length(tr)-length(vl) 1]));
    tt.test=cat(1,string(te),repmat([""],[length(tr)-length(te) 1]));
    writetable(tt,strcat('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/splits/bcl2_score_20x/splits_',num2str(i-1),'.csv'));
end

d=dir('leo_feats/cmyc/*.h5');
cellfun(@(x) strsplit(x,'.'),{d.name},'UniformOutput',false);
vertcat(ans{:});
ans(:,1);
slides=cellfun(@(x) str2num(x),ans);

for i=1:length(slides)
    te=slides(i);
    idxs=cat(2,1:i-1,i+1:length(slides));
    vl_idx=randsample(idxs,round(length(slides)*0.15));
    tr_idx=setdiff(idxs,vl_idx);
    tr=slides(tr_idx);
    vl=slides(vl_idx);
    
    tt=table;
    nums=strsplit(num2str(0:length(tr)-1),' ');
    tt.(" ")=string(nums)';
    tt.train=string(tr);
    tt.val=cat(1,string(vl),repmat([""],[length(tr)-length(vl) 1]));
    tt.test=cat(1,string(te),repmat([""],[length(tr)-length(te) 1]));
    writetable(tt,strcat('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/splits/cmyc_score_20x/splits_',num2str(i-1),'.csv'));
end