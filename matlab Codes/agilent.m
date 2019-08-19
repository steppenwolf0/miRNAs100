

dinfo = dir('*.txt');
great=[]
ids=[]
for K = 1 : length(dinfo)
  thisfilename = dinfo(K).name;
    fprintf( int2str( K ));
    fprintf( '"%s" \n',thisfilename);
    AGFEData = agferead(thisfilename)
    index=find(strcmp(AGFEData.ColumnNames,'gTotalGeneSignal'))
    gTotalGeneSignal=AGFEData.Data(:,index);
    names=AGFEData.IDs ;
    names = strrep(names,'miR', 'mir');
     names = strrep(names,'-star', '*');
     names = strrep(names,'_st', '');
    T = cell2table(names ,'VariableNames',{'Name'});
    T2 = array2table(gTotalGeneSignal);
    
    test=[T T2]
    matrixOut=test(~cellfun(@isempty, strfind(test.Name, 'hsa-')), :);
    matrixOut.Properties.VariableNames={'id','value'};
    statarray =  grpstats(matrixOut, 'id', {'sum'});
    statarray.Properties.VariableNames={'id','sum','value'};
    statarray.sum=[];
    great=[great statarray.value];
    ids=statarray.id;
    %fname = sprintf('%s_%d.csv','filename',K) 
end
ids = cell2table(ids ,'VariableNames',{'Name'});

great = array2table(great);
great=[ids great];

Xc = table2cell(great);
Xt = cell2table(Xc');

writetable(Xt,'GSE31164.csv','writevariablenames',0);