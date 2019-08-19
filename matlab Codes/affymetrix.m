ProbeStructure = affyrma('*', 'miRNA-2_0.cdf');

ExpressionMatrix=double(ProbeStructure);
test=ProbeStructure.RowNames;
test = strrep(test,'miR', 'mir');
test = strrep(test,'-star', '*');
test = strrep(test,'_st', '');
T = cell2table(test,'VariableNames',{'Name'});

%ExpressionMatrix(isnan(ExpressionMatrix))=0;

%ExpressionMatrix=zscore(ExpressionMatrix);

matrixOut=[T array2table(ExpressionMatrix)];
matrixOut=matrixOut(~cellfun(@isempty, strfind(matrixOut.Name, 'hp_hsa-')), :);

Xc = table2cell(matrixOut);
Xt = cell2table(Xc');

fname='GSE86277.csv';

writetable(Xt,fname,'writevariablenames',0);

MyFolderInfo = dir('*.CEL')
A_cell = struct2cell(MyFolderInfo);
A = cell2table(A_cell');
writetable(A(:,1),'labels.csv','writevariablenames',0);