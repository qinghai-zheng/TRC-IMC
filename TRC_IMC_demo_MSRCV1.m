clear,clc;
addpath('clusteringmeasure', 'tools');

paired_file_name = {'_miss0.1','_miss0.2', '_miss0.3', '_miss0.4', '_miss0.5', '_miss0.6', '_miss0.7', '_miss0.8', '_miss0.9'};
datadir= './datasets/';
data_name = 'MSRCV1';

fprintf('TRC-IMC with tensor nuclear norm on dataset: %s\n',data_name);
for missing_rate_num = 1:9
    datafile = [datadir, data_name, cell2mat(paired_file_name(missing_rate_num)), '.mat'];
    load(datafile);
    gt = truelabel{1};
    cls_num = length(unique(gt));
    % -----------------------------------------------
    lambda1_list = [0.1, 1, 10, 100, 1000, 10000];
    lambda2_list = [0.1, 1, 10, 100, 1000, 10000];
    dim_H = 150;
    % -----------------------------------------------
    fprintf('missing rates: %.1f \n', missing_rate_num*0.1)
    for lambda1_idx = 3:3 % 1:length(lambda1_list)
        for lambda2_idx = 3:3 % 1:length(lambda2_list)
            lambda1 = lambda1_list(lambda1_idx);
            lambda2 = lambda2_list(lambda2_idx);
            [Xo, Po, ~, Pm] = DataPreparing(data, index);
            num_views = length(data);
            for i = 1:num_views
                X{i} = Xo{i}*Po{i};
                X_ini{i} = Xo{i}*Po{i};
            end
            
            K = length(X);
            N = size(X{1},2);
            
            for k=1:K
                H{k} = zeros(dim_H,N);
                Z{k} = zeros(N,N);
                P{k} = zeros(size(X{k},1),dim_H);
                W1{k} = zeros(dim_H,N);
                W2{k} = zeros(N,N);
                C{k} = zeros(N,N);
                Q{k} = zeros(dim_H,N);
                E{k} = zeros(dim_H,N);
                Y1{k} = zeros(size(X{k},1),N);
                Y2{k} = zeros(dim_H,N);
            end
            
            w1 = zeros(dim_H*N*K,1);
            w2 = zeros(N*N*K,1);
            q = zeros(dim_H*N*K,1);
            g = zeros(N*N*K,1);
            
            dim1 = N;dim2 = N;dim3 = K; myNorm = 'tSVD_1';
            sX1 = [dim_H, N, K]; sX2 = [N, N, K];
            Isconverg = 0; epson = 1e-7; iter = 0;
            mu = 10e-5; max_mu = 10e10; pho_mu = 1.3;
            rho = 0.0001; max_rho = 10e12; pho_rho = 1.3;
            parOP = false; ABSTOL = 1e-6; RELTOL = 1e-4;
            
            while(Isconverg == 0)
                for k=1:K
                    % update H
                    H_a = mu*(P{k}'*P{k}) + (mu+rho)*eye(dim_H);
                    H_b = mu*(Z{k}*Z{k}'-Z{k}-Z{k}');
                    H_c = P{k}'*Y1{k}+mu*P{k}'*X{k}-Y2{k}+Y2{k}*Z{k}'-mu*E{k}*Z{k}'+mu*E{k}-W1{k}+rho*Q{k};
                    H{k} = lyap(H_a,H_b,H_c);
                    
                    % update P
                    P_a = X{k} + Y1{k}/mu;
                    P_b = H{k}*P_a';
                    [svd_U,~,svd_V] = svd(P_b,'econ');
                    P{k} = svd_V*svd_U';
                    
                    % update E
                    F = [];
                    for k_tmp1=1:K
                        F = [F; H{k_tmp1}-H{k_tmp1}*Z{k_tmp1}+Y2{k_tmp1}/mu];
                    end
                    [Econcat] = solve_l1l2(F,lambda2/mu);
                    dim_E_tmp = 0;
                    for k_tmp2=1:K
                        dim_E_tmp_last = dim_E_tmp;
                        dim_E_tmp = dim_E_tmp + dim_H;
                        E{k_tmp2} = Econcat(dim_E_tmp_last + 1:dim_E_tmp,:);
                    end
                    
                    % update Z
                    Z_a = rho*eye(N) + mu*H{k}'*H{k};
                    Z_b = rho*C{k} - W2{k} + H{k}'*Y2{k}+mu*H{k}'*(H{k}-E{k});
                    Z{k} = Z_a\Z_b;
                    
                    % updating Xm
                    Xm_tmp_1 = mu*Pm{k}*Pm{k}';
                    Xm_tmp_2 = (mu*P{k}*H{k} - Y1{k} - mu*Xo{k}*Po{k})*Pm{k}';
                    Xm{k} = Xm_tmp_2/Xm_tmp_1;
                    X{k} = X_ini{k} + NormalizeData(Xm{k})*Pm{k};
                    
                    %update Y1 Y2
                    Y1{k} = Y1{k} + mu*(X{k}-P{k}*H{k});
                    Y2{k} = Y2{k} + mu*(H{k}-H{k}*Z{k}-E{k});
                end
                % update Q
                H_tensor = cat(3, H{:,:});
                W1_tensor = cat(3, W1{:,:});
                h = H_tensor(:);
                w1 = W1_tensor(:);
                [q, ~] = wshrinkObj(h + 1/rho*w1,1/rho,sX1,0,3);
                Q_tensor = reshape(q, sX1);
                
                % update C
                Z_tensor = cat(3, Z{:,:});
                W2_tensor = cat(3, W2{:,:});
                z = Z_tensor(:);
                w2 = W2_tensor(:);
                
                [c, ~] = wshrinkObj(z + 1/rho*w2,lambda1/rho,sX2,0,3);
                C_tensor = reshape(c, sX2);
                
                %5 update W1 W2
                w1 = w1 + rho*(h - q);
                w2 = w2 + rho*(z - c);
                
                Isconverg = 1;
                for k=1:K
                    if (norm(X{k}-P{k}*H{k},inf)>epson)
                        history.norm_X = norm(X{k}-P{k}*H{k},inf);
                        Isconverg = 0;
                    end
                    if (norm(H{k}-H{k}*Z{k}-E{k},inf)>epson)
                        history.norm_H = norm(H{k}-H{k}*Z{k}-E{k},inf);
                        Isconverg = 0;
                    end
                    
                    Q{k} = Q_tensor(:,:,k);
                    W1_tensor = reshape(w1, sX1);
                    W1{k} = W1_tensor(:,:,k);
                    C{k} = C_tensor(:,:,k);
                    W2_tensor = reshape(w2, sX2);
                    W2{k} = W2_tensor(:,:,k);
                    if (norm(H{k}-Q{k},inf)>epson)
                        history.norm_H_Q = norm(H{k}-Q{k},inf);
                        Isconverg = 0;
                    end
                    if (norm(Z{k}-C{k},inf)>epson)
                        history.norm_Z_C = norm(Z{k}-C{k},inf);
                        Isconverg = 0;
                    end
                end
                
                if (iter>100)  % (iter>200)
                    Isconverg  = 1;
                end
                iter = iter + 1;
                mu = min(mu*pho_mu, max_mu);
                rho = min(rho*pho_rho, max_rho);
            end
            S = 0;
            for k=1:K
                S = S + abs(Z{k})+abs(Z{k}');
            end
            
            Cresults = SpectralClustering(S,cls_num);
            [~, nmi, ~] = compute_nmi(gt,Cresults);
            ACC = Accuracy(Cresults,double(gt));
            [fscore,precision_score,~] = compute_f(gt,Cresults);
            
            fprintf('\t dim_H: %d, lambda1: %f, lambda2: %f \n',dim_H,lambda1, lambda2);
            fprintf('\t NMI: %f, ACC: %f, F-Score: %f, Precision: %f\n',nmi,ACC,fscore,precision_score);
        end
    end
end
fprintf('----------------------------------------------------------------------------\n');
