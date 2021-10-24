%MAIN FILE - RUN THIS
%-0 for regression, 1 for classification
binary_variable = 0;
decision_tree_learning(binary_variable);
binary_variable = 1;
decision_tree_learning(binary_variable);


function decision_tree_learning(binary_variable)

    if binary_variable == 0
        regression_tree;
    else
        c_table = readtable('classification_dataset.csv', 'ReadVariableNames',true, "PreserveVariableNames",true);
        columnIndicesToDelete = [1,2,17,19,20,21,22,23];
        c_table(:,columnIndicesToDelete) = [];
        c_table.Properties.VariableNames = {'SR', 'NR', 'TR', 'VR', 'SUR1', 'SUR2', 'SUR3', 'UR', 'FR', 'OR', 'RR', 'BR', 'MR', 'CR', 'Brown frogs'};
        c_array = table2array(c_table);
        attribute_names = c_table.Properties.VariableNames;
        c_tree = classification_tree(c_array, attribute_names);
     end

        


end



