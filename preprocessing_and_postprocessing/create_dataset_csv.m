t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo312_tma/leo312_clinical.csv');

rd='/isilon/datalake/cialab/original/cialab/image_database/d00124/';
case_ids={};
slide_ids={};
efs_stats={};
for i=1:size(t,1)
    slides=strsplit(t.bcl2_files{i},',');
    for j=1:length(slides)
        if isempty(slides{j}) || isnan(t.efs_stat(i))
            continue;
        end
        fn=strcat(rd,slides{j});
        if ~exist(fn,'file')
            fprintf('%s not found\n',slides{j});
            continue;
        end
        system(strjoin(["ln -s",fn,strcat("bcl2/",slides{j})]," "));
        case_id=strsplit(slides{j},'_'); case_id=case_id{1};
        case_ids=cat(1,case_ids,case_id);
        slide_id=strsplit(slides{j},'.'); slide_id=slide_id{1};
        slide_ids=cat(1,slide_ids,slide_id);
        if t.efs_stat(i)==2
            efs_stats=cat(1,efs_stats,"yes");
        else
            efs_stats=cat(1,efs_stats,"no");
        end
    end
end
tt=table;
tt.case_id=case_ids;
tt.slide_id=slide_ids;
tt.label=efs_stats;
writetable(tt,'/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/dataset_csv/bcl2_efs.csv');

rd='/isilon/datalake/cialab/original/cialab/image_database/d00124/';
case_ids={};
slide_ids={};
stage_grps=[];
for i=1:size(t,1)
    slides=strsplit(t.bcl2_files{i},',');
    for j=1:length(slides)
        if isempty(slides{j}) || isnan(t.stage_grp(i))
            continue;
        end
        fn=strcat(rd,slides{j});
        if ~exist(fn,'file')
            fprintf('%s not found\n',slides{j});
            continue;
        end
        system(strjoin(["ln -s",fn,strcat("bcl2/",slides{j})]," "));
        case_id=strsplit(slides{j},'_'); case_id=case_id{1};
        case_ids=cat(1,case_ids,case_id);
        slide_id=strsplit(slides{j},'.'); slide_id=slide_id{1};
        slide_ids=cat(1,slide_ids,slide_id);
        if t.stage_grp(i)==1
            stage_grps=cat(1,stage_grps,"yes");
        else
            stage_grps=cat(1,stage_grps,"no");
        end
    end
end
tt=table;
tt.case_id=case_ids;
tt.slide_id=slide_ids;
tt.label=stage_grps;
writetable(tt,'/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/dataset_csv/bcl2_stage.csv');

rd='/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo/';
case_ids={};
slide_ids={};
efs_stats=[];
for i=1:size(t,1)
    slides=strsplit(t.cmyc_files{i},',');
    for j=1:length(slides)
        if isempty(slides{j}) || isnan(t.efs_stat(i))
            continue;
        end
        fn=strcat(rd,slides{j});
        if ~exist(fn,'file')
            fprintf('%s not found\n',slides{j});
            continue;
        end
        system(strjoin(["ln -s",strcat('"',fn,'"'),strcat("cmyc/",slides{j})]," "));
        case_id=strsplit(slides{j},'_'); case_id=case_id{1};
        case_ids=cat(1,case_ids,case_id);
        slide_id=strsplit(slides{j},'.'); slide_id=slide_id{1};
        slide_ids=cat(1,slide_ids,slide_id);
        if t.efs_stat(i)==2
            efs_stats=cat(1,efs_stats,"yes");
        else
            efs_stats=cat(1,efs_stats,"no");
        end
    end
end
tt=table;
tt.case_id=case_ids;
tt.slide_id=slide_ids;
tt.label=efs_stats;
writetable(tt,'/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/dataset_csv/cmyc_efs.csv');

rd='/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo/';
case_ids={};
slide_ids={};
stage_grps=[];
for i=1:size(t,1)
    slides=strsplit(t.cmyc_files{i},',');
    for j=1:length(slides)
        if isempty(slides{j}) || isnan(t.stage_grp(i))
            continue;
        end
        fn=strcat(rd,slides{j});
        if ~exist(fn,'file')
            fprintf('%s not found\n',slides{j});
            continue;
        end
        system(strjoin(["ln -s",strcat('"',fn,'"'),strcat("cmyc/",slides{j})]," "));
        case_id=strsplit(slides{j},'_'); case_id=case_id{1};
        case_ids=cat(1,case_ids,case_id);
        slide_id=strsplit(slides{j},'.'); slide_id=slide_id{1};
        slide_ids=cat(1,slide_ids,slide_id);
        if t.stage_grp(i)==1
            stage_grps=cat(1,stage_grps,"yes");
        else
            stage_grps=cat(1,stage_grps,"no");
        end
    end
end
tt=table;
tt.case_id=case_ids;
tt.slide_id=slide_ids;
tt.label=stage_grps;
writetable(tt,'/isilon/datalake/cialab/scratch/cialab/tet/python/media/CLAM/dataset_csv/cymc_stage.csv');
