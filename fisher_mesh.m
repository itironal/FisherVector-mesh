
%This code uses mesh weights to extract fisher vector representations and
%computes classification using SVM. To run this code, you need to install
%VLfeat for MATLAB from http://www.vlfeat.org/install-matlab.html ,
%uncomment the line below and update the vlfeat path.
%run('vlfeat-0.9.20/toolbox/vl_setup')
%
%To use this code you need to install liblinear, uncomment the line below
%and update the liblinear path with your own liblinear path.
%addpath(genpath('E:\liblinear-2.1'));

p = 40;
lambda = 512;
number_of_subjects = 100 ;%numel(subjects);
subject_sample_size = 7;
foldNo = 10; % We perform 10-fold CV. Update foldNo for different number of folds
pca_reduced_dim_size = 90;
all_labels = repmat(1:7 ,1, number_of_subjects)';
numRegs = 90;
load(['mesh_weights/p' num2str(p) '/lambda' num2str(lambda) '/weights.mat']);

for i = 1:foldNo
    
    te_inds = (i-1)*subject_sample_size * foldNo + 1: i *subject_sample_size * foldNo;
    tr_inds = setdiff(1: number_of_subjects*subject_sample_size, te_inds);
    
    patches_all = reshape(all_subjects_a', numRegs, size(all_subjects_a,1)*numRegs)';
    
    patches_tr = reshape(all_subjects_a(tr_inds,:)', numRegs, size(all_subjects_a(tr_inds,:),1)*numRegs)';
    patches_te = reshape(all_subjects_a(te_inds,:)', numRegs, size(all_subjects_a(te_inds,:),1)*numRegs)';
    
    [COEFF, SCORE] = princomp(patches_tr); % use only training part to perform PCA
    
    new_data_tr = (COEFF(:,1:pca_reduced_dim_size)' *  patches_tr');
    new_data_all = (COEFF(:,1:pca_reduced_dim_size)' *  patches_all')';
    for numClusters = 20:20:120
        
        disp(['p = ' num2str(p) ...
            '  lambda = ' num2str(lambda)]);
        disp(['numClusters = ' num2str(numClusters), ...
            '  fold = ' num2str(i)]);
        
        [means,covariances,priors,ll,posteriors] = vl_gmm(new_data_tr, numClusters);
        
        onerun.means = means; % initMeans;
        onerun.covariances = covariances; %initCovariances;
        onerun.priors = priors; %initPriors;
        
        all_encodings = zeros(subject_sample_size * number_of_subjects, pca_reduced_dim_size * numClusters*2);
        for k = 1: subject_sample_size * number_of_subjects
            
            dataToBeEncoded = new_data_all((k-1)*numRegs  + 1 : k * numRegs ,:);
            all_encodings(k,:) = vl_fisher(dataToBeEncoded', means, covariances, priors, 'Improved');
        end
        
        
        model = train_linear(all_labels(tr_inds), sparse(all_encodings(tr_inds,:)), ['-c 1',' -s 2 -q heart_scale ']);
        [predicted_label,accuracy,prob] = predict_linear(all_labels(te_inds), sparse(all_encodings(te_inds,:)), model, '-b 1');
        
        
        gt_label = all_labels(te_inds);
        mkdir(['fisher_results/p' num2str(p) '/lambda' num2str(lambda) ...
            '/cl' num2str(numClusters) '/fold' num2str(i)]);
        save(['fisher_results/p' num2str(p) '/lambda' num2str(lambda) ...
            '/cl' num2str(numClusters) '/fold' num2str(i) '/res.mat'], 'accuracy', 'predicted_label', 'gt_label', 'onerun');
    end
    
end


