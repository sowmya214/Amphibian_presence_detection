function tree = classification_tree(array, attribute_names)

tree = decision_tree_learning(array(:, [1:14]), array(:, [15]), attribute_names);
DrawDecisionTree(tree)
%[accuracy, precision, recall] = calculate_accuracy(array(:, [1:14]), array(:, [15]), attribute_names, result)
avg_eval = cross_validation(array(:, [1:15]), attribute_names);

end

function decision_tree = decision_tree_learning(features,labels, attribute_names);

    %creating empty tree
    tree = struct('op', [], 'kids', [], 'attribute_index', [], 'attribute', [], 'threshold', [], 'class', []);
    
    %if the labels in the node are the same, node is a leaf node
    if range(labels) == 0
        tree.op = "";
        tree.class = labels(1);
        decision_tree = tree;    
    else
        %[best_attribute, best_threshold] <- CHOOSEE-ATTRIBUTE(features,
        %targets)
        [best_attribute, best_threshold,examples_l,targets_l,examples_r,targets_r] = choose_attribute(features, labels);
        
        %tree <- tree with root decision attribute best_attribute and
        %threshold best_threshold 
        tree.attribute_index = best_attribute;
        tree.attribute = attribute_names(best_attribute);
        tree.threshold = best_threshold;
        tree.op = tree.attribute + " < " + tree.threshold;
        tree.class = "";
        
        %subtree <- DECISION_TREE_LEARNING(examples_l, targets_l)
        tree.kids{end+1} = decision_tree_learning(examples_l, targets_l, attribute_names);
        %subtree <- DECISION_TREE_LEARNING(examples_l, targets_l)
        tree.kids{end+1} = decision_tree_learning(examples_r, targets_r, attribute_names);
        decision_tree = tree;         
    end
end

%FIND ESTIMATE ON INFORMATION CONTAINED IN A CORRECT ANSWER
function information = get_information(data)
   p = sum(data)/length(data);
   n = 1 - p;
   information = (-p * log2(p)) -(n * log2(n));
end

%SPLIT DATA INTO LEFT AND RIGHT SUB SET SO THAT IT CAN BE PASSED INTO
%SUBTREES
function [left_attributes, right_attributes, left_labels, right_labels] = split(data, threshold, features, labels)
    left = features(:,data)<threshold;
    right = features(:,data)>=threshold;
    left_attributes = features(left,:);
    right_attributes = features(right,:);
    left_labels = labels(left,:);
    right_labels = labels(right,:);
end

%CHOOSE-ATTRIBUTE FUNCTION
function [best_attribute, best_threshold, examples_l, targets_l, examples_r, targets_r] = choose_attribute(features, labels)

    best_attribute = 0;
    best_threshold = 0;
    best_gain = 0;
    gain_present = false;
    total_information = get_information(labels); 
    
    % Looping through each feature
    for i = 1 : size(features, 2) - 1
        
        unique_subset = unique(features(:, i), 'rows');
        current_attribute = i;
      
        % checking which threshold/unique value of attribute works best for current feature
        for j = 1 : length(unique_subset) - 1
         
            current_threshold = ((unique_subset(j+1) - unique_subset(j))/2) + unique_subset(j);
            %Splitting data in left and right subsets (sub trees) based on threshold
            [left_features_subset, right_features_subset, left_labels_subset, right_labels_subset] = split(current_attribute, current_threshold, features, labels);
 
            left_information = get_information(left_labels_subset);
            right_information = get_information(right_labels_subset);
            
            if isnan(left_information)
               left_information = 0;
            end
            
            if isnan(right_information)
               right_information = 0;
            end
            
            remainder = length(left_features_subset)/length(labels)*left_information+length(right_features_subset)/length(labels)*right_information;
            gain = total_information - remainder;
                  
            if(gain > best_gain)
                best_threshold = current_threshold;
                best_attribute = current_attribute;
                best_gain = gain;
                examples_l = left_features_subset;
                targets_l = left_labels_subset;
                examples_r = right_features_subset;
                targets_r = right_labels_subset;
                gain_present = true;
            end
        end
    end
    %in the case that there is no gain, but attribute still needs to be
    %chosen for split to happen
    if(gain_present == false)
        best_threshold = current_threshold;
        best_attribute = current_attribute;
        examples_l = left_features_subset;
        targets_l = left_labels_subset;
        examples_r = right_features_subset;
        targets_r = right_labels_subset;
    end
end

%ENTER A SPECIFIC NUMBER i CORRESPONDING TO A SAMPLE IN DATASET TO GET A
%PREDICTION FOR THAT SAMPLE
function prediction = get_prediction(i, features, attribute_names, tree)   
    pred = tree.class;
    %if there is no prediction specified at current node
    if strcmp(pred,"")
        current_attribute = features(i, tree.attribute_index);
        if current_attribute < tree.threshold
            prediction = get_prediction(i, features, attribute_names, tree.kids{1});
        else
             prediction = get_prediction(i, features, attribute_names, tree.kids{2});
        end
    else 
        prediction = pred;
    end
end

%CALCULATES ACCURACY, PRECISION AND RECALL 
function [accuracy, precision, recall, f1] = calculate_accuracy(features, labels, attribute_names, tree)
    correct = 0;
    true_positives = 0;
    false_positives = 0;
    false_negatives = 0;
    for i=1: length(features)
        prediction = get_prediction(i, features, attribute_names, tree);
        if prediction == labels(i)
            correct = correct + 1;
        end
        if prediction == 1
            if prediction == labels(i)
                true_positives = true_positives + 1;
            else
                false_positives = false_positives + 1;
            end
        else 
            if prediction ~= labels(i)
                false_negatives = false_negatives + 1;
            end
        end
    end
    accuracy = correct/(length(features)) * 100;
    precision = true_positives/(true_positives + false_positives);
    recall = true_positives/(true_positives + false_negatives);
    f1 = 2*((precision*recall)/(precision+recall));
end

%10 FOLD CROSS VALIDATION
function result = cross_validation(data, attribute_names)
    shuffledArray = data(randperm(180),:);
    folds = mat2cell(shuffledArray, size(shuffledArray,1)/10.*ones(10,1), 15);
    avg_accuracy = 0;
    avg_precision = 0;
    avg_recall = 0;
    avg_f1 = 0;
    
    for i=1 : size(folds, 1)
        test_fold = folds{i};        
        training_folds = folds{1:9};
        tmp_tree = decision_tree_learning(training_folds(:, [1:14]), training_folds(:, [15]), attribute_names);
        [fold_accuracy, fold_precision, fold_recall, fold_f1] = calculate_accuracy(test_fold(:, [1:14]), test_fold(:, [15]), attribute_names, tmp_tree);
        result = strcat('testing on fold: ', num2str(i), '\n');
        fprintf(result);
        result = strcat('accuracy: ', num2str(fold_accuracy), ', precision: ', num2str(fold_precision), ', recall: ', num2str(fold_recall),', f1: ', num2str(fold_f1), '\n\n');
        fprintf(result);
        avg_accuracy = avg_accuracy + fold_accuracy;
        avg_precision = avg_precision + fold_precision;
        avg_recall = avg_recall + fold_recall;
        avg_f1 = avg_f1 + fold_f1;
    end
    
    avg_accuracy = avg_accuracy/10;
    avg_precision = avg_precision/10;
    avg_recall = avg_recall/10;
    avg_f1 = avg_f1/10;
    result = strcat('AVERAGE ACCURACY, PRECISION, RECALL, F1 across all folds: ', num2str(avg_accuracy), ', ',num2str(avg_precision), ', ', num2str(avg_recall), ', ',num2str(avg_f1));
    fprintf(result);
end
