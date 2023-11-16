d=dir('leo_feats_tmas/bcl2/40x/*.h5');
cellfun(@(x) strsplit(x,'.'),{d.name},'UniformOutput',false);
vertcat(ans{:});
ans(:,1);
string(ans);
t=readtable('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/dataset_csv/leo_bcl2_score.csv');
h5_names=ans;
slide_labels=[];
for i=1:length(h5_names)
    s=strsplit(h5_names(i),'_');
    s=str2num(s{1});
    idx=find(t.slide_id==s);
    slide_labels(i)=t.label(idx);
end
tt=table;
tt.slide_id=h5_names;
tt.case_id=h5_names;
tt.label=slide_labels';
writetable(tt,'/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/dataset_csv/leo_bcl2_score_40x_tmas.csv');

tt=table;
tt.train=h5_names;
tt.val=h5_names;
tt.test=h5_names;
for i=0:9
    writetable(tt,strcat('/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/splits/bcl2_score_40x_tmas/splits_',num2str(i),'.csv'));
end