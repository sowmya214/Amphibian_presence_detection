%RUN THIS FOR DECISION TREE

function decision_tree_learning(binary_variable)
    c_table = readtable('dataset.csv', 'ReadVariableNames',true, "PreserveVariableNames",true);
    columnIndicesToDelete = [1,2,17,19,20,21,22,23];
    c_table(:,columnIndicesToDelete) = [];
    c_table.Properties.VariableNames = {'SR', 'NR', 'TR', 'VR', 'SUR1', 'SUR2', 'SUR3', 'UR', 'FR', 'OR', 'RR', 'BR', 'MR', 'CR', 'Brown frogs'};
    c_array = table2array(c_table);
    attribute_names = c_table.Properties.VariableNames;
    c_tree = classification_tree(c_array, attribute_names);
end



