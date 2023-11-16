rd='/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo/';
t=readtable('/isilon/datalake/cialab/original/cialab/image_database/d00134/Whole Slides Image/leo312_tma/leo312_clinical.csv');

ps=224;
for i=1:size(t,1)
    slides=t.bcl2_files{i};
    if isempty(slides)
        continue;
    end

    slides=strsplit(slides,',');
    for s=1:length(slides)
        fp=openslide_open(strcat(rd,slides{s}));
        [w,h]=openslide_read_region(fp);
