%% PENALIZED MAXIMUM LIKELIHOOD ESTIMATION
% Univariate simulation study
%
% For each simulated sample and each number of inner knots,
% the penalization parameter is selected by K-fold cross-validation.
%
% Required auxiliary functions:
%   ML_1D_grad.m
%   trapz_weights_1D.m

clear;
clc;
close all;

%% =========================================================
% SETTINGS
%% =========================================================

density = 1;            % 1 = Normal, 2 = Bimodal, 3 = Trimodal

N = 3000;               % sample size
K = 5;                  % number of cross-validation folds

n_simulations = 100;    % number of simulation runs

g_grid = 1:10;          % numbers of inner knots
rho_grid = logspace(-4,1,50);

k_deg = 2;              % spline degree
d = 1;                  % difference order

n_grid = 5000;          % grid size for numerical integration

%% =========================================================
% OPTIMIZATION OPTIONS
%% =========================================================

opts = optimoptions('fminunc', ...
    'Algorithm','quasi-newton', ...
    'SpecifyObjectiveGradient',true, ...
    'Display','off', ...
    'MaxFunctionEvaluations',5e4, ...
    'MaxIterations',2e3);

%% =========================================================
% RESULTS
%% =========================================================

results = table();

%% =========================================================
% SIMULATION
%% =========================================================

for isim = 1:n_simulations

    fprintf('\nSimulation %d/%d\n', ...
        isim,n_simulations);

    %% -----------------------------------------------------
    % Generate data
    %% -----------------------------------------------------

    switch density

        case 1
            density_name = "Normal";

            mu = 0;
            sigma = 1;

            x = normrnd(mu,sigma,[N,1]);

            a = -4;
            b = 4;

        case 2
            density_name = "Bimodal";

            mu1 = -2;
            mu2 = 2;

            sigma1 = 0.4;
            sigma2 = 0.7;

            u = rand(N,1);
            x = zeros(N,1);

            idx1 = u < 0.5;
            idx2 = ~idx1;

            x(idx1) = ...
                normrnd(mu1,sigma1,[sum(idx1),1]);

            x(idx2) = ...
                normrnd(mu2,sigma2,[sum(idx2),1]);

            a = -4;
            b = 4;

        case 3
            density_name = "Trimodal";

            mu1 = -3;
            mu2 = 0;
            mu3 = 2;

            sigma1 = 0.3;
            sigma2 = 0.2;
            sigma3 = 0.2;

            n1 = floor(N/3);
            n2 = floor(N/3);
            n3 = N-n1-n2;

            x = [
                normrnd(mu1,sigma1,[n1,1]);
                normrnd(mu2,sigma2,[n2,1]);
                normrnd(mu3,sigma3,[n3,1])
            ];

            a = -4;
            b = 3;

        otherwise
            error('Unknown density.');
    end

    %% -----------------------------------------------------
    % Evaluation grid
    %% -----------------------------------------------------

    xq = linspace(a,b,n_grid)';

    %% -----------------------------------------------------
    % True density
    %% -----------------------------------------------------

    switch density

        case 1
            f_true = ...
                normpdf(xq,mu,sigma);

        case 2
            f_true = ...
                0.5*normpdf(xq,mu1,sigma1) + ...
                0.5*normpdf(xq,mu2,sigma2);

        case 3
            f_true = ...
                (1/3)*normpdf(xq,mu1,sigma1) + ...
                (1/3)*normpdf(xq,mu2,sigma2) + ...
                (1/3)*normpdf(xq,mu3,sigma3);
    end

    %% -----------------------------------------------------
    % clr representation of the true density
    %% -----------------------------------------------------

    clr_true = ...
        log(f_true) ...
        - trapz(xq,log(f_true))/(b-a);

    %% -----------------------------------------------------
    % Cross-validation partition
    %% -----------------------------------------------------

    cv = cvpartition(N,'KFold',K);

    %% =====================================================
    % LOOP OVER THE NUMBER OF INNER KNOTS
    %% =====================================================

    for g = g_grid

        fprintf('   g = %d\n',g);

        %% -------------------------------------------------
        % ZB-spline basis
        %% -------------------------------------------------

        lambda = linspace(a,b,g+2);

        Lambda = ...
            augknt(lambda,k_deg+1);

        Zsys = [];

        for i = 1:length(Lambda)-k_deg-2

            B = ...
                spmak( ...
                    Lambda(i:i+k_deg+2), ...
                    1);

            Zsys = ...
                [Zsys fnder(B)]; %#ok<AGROW>
        end

        m = length(Zsys);

        %% -------------------------------------------------
        % Collocation matrix at observations
        %% -------------------------------------------------

        Zx = zeros(N,m);

        for i = 1:m
            Zx(:,i) = fnval(Zsys(i),x)';
        end

        %% -------------------------------------------------
        % Collocation matrix on the integration grid
        %% -------------------------------------------------

        Zq = zeros(length(xq),m);

        for i = 1:m
            Zq(:,i) = fnval(Zsys(i),xq)';
        end

        %% -------------------------------------------------
        % Penalty matrix
        %% -------------------------------------------------

        D = diff(eye(m),d);
        P = D'*D;

        %% -------------------------------------------------
        % Select rho by K-fold cross-validation
        %% -------------------------------------------------

        cv_score = zeros(length(rho_grid),1);

        for ir = 1:length(rho_grid)

            rho = rho_grid(ir);

            fold_score = zeros(K,1);

            for kfold = 1:K

                idx_train = training(cv,kfold);
                idx_test  = test(cv,kfold);

                c_train = ...
                    sum(Zx(idx_train,:),1)';

                c_test = ...
                    sum(Zx(idx_test,:),1)';

                N_train = sum(idx_train);
                N_test  = sum(idx_test);

                objective = @(z) ...
                    ML_1D_grad( ...
                        z, ...
                        c_train, ...
                        N_train, ...
                        Zq, ...
                        xq, ...
                        P, ...
                        rho);

                z_hat = ...
                    fminunc( ...
                        objective, ...
                        zeros(m,1), ...
                        opts);

                %% validation negative log-likelihood

                s = Zq*z_hat;

                smax = max(s);

                log_integral = ...
                    smax + ...
                    log( ...
                        trapz( ...
                            xq, ...
                            exp(s-smax)));

                fold_score(kfold) = ...
                    -c_test'*z_hat ...
                    + N_test*log_integral;
            end

            cv_score(ir) = mean(fold_score);
        end

        %% -------------------------------------------------
        % Optimal penalization parameter
        %% -------------------------------------------------

        [cv_min,best_idx] = min(cv_score);

        rho_opt = rho_grid(best_idx);

        %% -------------------------------------------------
        % Final fit
        %% -------------------------------------------------

        c = sum(Zx,1)';

        objective = @(z) ...
            ML_1D_grad( ...
                z, ...
                c, ...
                N, ...
                Zq, ...
                xq, ...
                P, ...
                rho_opt);

        z_final = ...
            fminunc( ...
                objective, ...
                zeros(m,1), ...
                opts);

        %% -------------------------------------------------
        % Estimated clr density
        %% -------------------------------------------------

        s_est = Zq*z_final;

        %% -------------------------------------------------
        % Estimated probability density
        %% -------------------------------------------------

        smax = max(s_est);

        f_est = exp(s_est-smax);

        f_est = ...
            f_est/trapz(xq,f_est);

        %% -------------------------------------------------
        % Normalized integrated squared error
        %% -------------------------------------------------

        NISE = ...
            trapz( ...
                xq, ...
                (clr_true-s_est).^2) ...
            /(b-a);

        %% -------------------------------------------------
        % Store results
        %% -------------------------------------------------

        new_row = table( ...
            isim, ...
            density_name, ...
            N, ...
            K, ...
            g, ...
            rho_opt, ...
            cv_min, ...
            NISE, ...
            'VariableNames', { ...
                'simulation', ...
                'density', ...
                'N', ...
                'K', ...
                'g', ...
                'rho_opt', ...
                'CV', ...
                'NISE'});

        results = ...
            [results; new_row]; %#ok<AGROW>

        fprintf( ...
            '      rho = %.4g, CV = %.4f, NISE = %.4g\n', ...
            rho_opt,cv_min,NISE);
    end
end

%% =========================================================
% SUMMARY
%% =========================================================

summary = ...
    groupsummary( ...
        results, ...
        'g', ...
        {'mean','median','std'}, ...
        {'rho_opt','CV','NISE'});

disp(summary);

%% =========================================================
% SAVE RESULTS
%% =========================================================

writetable( ...
    results, ...
    sprintf( ...
        'simulation_1D_%s_N%d_K%d.csv', ...
        density_name,N,K));

writetable( ...
    summary, ...
    sprintf( ...
        'simulation_1D_%s_N%d_K%d_summary.csv', ...
        density_name,N,K));