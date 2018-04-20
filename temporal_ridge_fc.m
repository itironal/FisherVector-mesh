

function [a_tr, tr_error]=temporal_ridge_fc(corrs,p_value,tr_all,lambda,duration)

% pos: voxel_number by 3 matrix 
% p_value: the number of neighbor voxels (p in the algorithm )
% tr_all, te_all: train and test data (N by voxel_number matrix, N:sample size

a_tr=[];
tr_error =[];

for i=1:size(tr_all,2) 

   [neighbor_index] = find_nn_corr(corrs, p_value, i);

    a1=[];
    error = [];
    for j = 1:(size(tr_all,1)/duration)
        y=tr_all(duration*(j-1) + 1 : duration*(j-1) +duration,i);
        X=tr_all(duration*(j-1) + 1 : duration*(j-1) +duration , neighbor_index{1,1});
        
        theta = ridge(y,X,lambda);
        err = sum((X*theta - y).*(X*theta - y));
        tempp = zeros(1,size(tr_all,2));
        tempp(neighbor_index{1,1}) = theta';
        a1 = [a1;tempp];
        error = [error err];
    end
    tr_error = [tr_error;error];

    a_tr=[a_tr a1];

end
   