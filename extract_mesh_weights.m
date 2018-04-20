%For HCP dataset data path contains .mat files containing data of each
%subject separately (subj1.mat, subj2.mat, ...). Each of these .mat files
%contains a variable sub_data of size M x N where M represents the number
%of time instants and N represents the number of regions. For HCP dataset,
%M = 1940 (sum of all values in durations) and N = 116. 
%
%This code extracts and saves mesh weights and class labels of each trial as output.


all_files = dir('data/*.mat');

durations = [176, 253, 316, 284, 232, 274, 405];
subject_sample_size = 7;

data_path = 'data';
p = 40;
lambda = 512;

all_subjects_a = [];
mvpa_data = [];
labels = [];
mkdir(fullfile('mesh_weights', ['p' num2str(p)], ['lambda' num2str(lambda)]));
mkdir(fullfile('labels', ['p' num2str(p)], ['lambda' num2str(lambda)]));

for subj = 1:numel(all_files)
    load(fullfile('data', all_files(subj).name));
 
    all_a = [];
    ind_begin = 1;
    
    for expr = 1:7
        each_data = sub_data(ind_begin: ind_begin + durations(expr)-1 ,[1:8, 27:108]); %exclude regions in Cerebellum and Vermis
        [a_data, errors]=temporal_ridge_fc(corr(each_data),p, each_data, lambda_vals(lambda), floor(durations(expr)));

        ind_begin = ind_begin + durations(expr);
        all_a = [all_a;a_data];
        labels = [labels;repmat(expr,size(a_data,1),1)];
    end
    all_subjects_a = [all_subjects_a;all_a];
end

save(fullfile('mesh_weights', ['p' num2str(p)], ['lambda' num2str(lambda)], 'weights.mat'), 'all_subjects_a');
save(fullfile('labels', ['p' num2str(p)], ['lambda' num2str(lambda)], 'labels.mat'), 'labels');

