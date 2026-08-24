%% PENALIZED MAXIMUM LIKELIHOOD ESTIMATION
% Univariate application to lead (Pb) concentration data
%
% This script reproduces the univariate geochemical application
% described in the paper. For each Czech administrative district
% and each number of inner knots, the penalization parameter is
% selected by K-fold cross-validation.
%
% Required auxiliary functions:
%   ML_1D_grad.m
%   trapz_weights_1D.m

clear;
clc;
close all;

%% =========================================================
% LOAD DATA
%% =========================================================

opts = detectImportOptions('data.csv','Delimiter',';');

opts = setvartype(opts,'Pb','string');

data = readtable('data.csv',opts);

data.Pb = ...
    str2double( ...
        replace(data.Pb,',','.'));

%% =========================================================
% COMMON DOMAIN
%% =========================================================

a = 1.358;
b = 5.716;

%% =========================================================
% SETTINGS
%% =========================================================

k_deg = 3;              % spline degree
d = 1;                  % difference order

g_grid = 1:10;          % numbers of inner knots
rho_grid = logspace(-4,1,50);

K = 5;                  % number of cross-validation folds

n_grid = 5000;          % grid size for numerical integration

%% =========================================================
% OPTIMIZATION OPTIONS
%% =========================================================

opts_fmin = optimoptions( ...
    'fminunc', ...
    'Algorithm','quasi-newton', ...
    'SpecifyObjectiveGradient',true, ...
    'Display','off', ...
    'MaxFunctionEvaluations',5e4, ...
    'MaxIterations',2e3);

%% =========================================================
% ADMINISTRATIVE DISTRICTS
%% =========================================================

regions = ...
    string(unique(data.NAZEV));

sortkey = lower(regions);

sortkey = ...
    replace(sortkey,'ch','h~');

[~,idx] = sort(sortkey);

regions = regions(idx);

%% =========================================================
% RESULTS
%% =========================================================

results = table();

%% =========================================================
% LOOP OVER DISTRICTS
%% =========================================================

for iregion = 1:length(regions)

    region = ...
        regions(iregion);

    mask = ...
        strcmpi(data.NAZEV,region);

    x = ...
        data.Pb(mask);

    x = ...
        x(~isnan(x));

    N = ...
        length(x);

    fprintf( ...
        '\nRegion: %s (%d/%d), N = %d\n', ...
        region, ...
        iregion, ...
        length(regions), ...
        N);

    %% -----------------------------------------------------
    % Evaluation grid
    %% -----------------------------------------------------

    xq = ...
        linspace(a,b,n_grid)';

    %% -----------------------------------------------------
    % Cross-validation partition
    %% -----------------------------------------------------

    cv = ...
        cvpartition(N,'KFold',K);

    %% =====================================================
    % LOOP OVER THE NUMBER OF INNER KNOTS
    %% =====================================================

    for g = g_grid

        fprintf('   g = %d\n',g);

        %% -------------------------------------------------
        % ZB-spline basis
        %% -------------------------------------------------

        lambda = ...
            linspace(a,b,g+2);

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

        m = ...
            length(Zsys);

        %% -------------------------------------------------
        % Collocation matrix at observations
        %% -------------------------------------------------

        Zx = ...
            zeros(N,m);

        for i = 1:m

            Zx(:,i) = ...
                fnval(Zsys(i),x)';
        end

        %% -------------------------------------------------
        % Collocation matrix on the integration grid
        %% -------------------------------------------------

        Zq = ...
            zeros(length(xq),m);

        for i = 1:m

            Zq(:,i) = ...
                fnval(Zsys(i),xq)';
        end

        %% -------------------------------------------------
        % Penalty matrix
        %% -------------------------------------------------

        D = ...
            diff(eye(m),d);

        P = ...
            D'*D;

        %% -------------------------------------------------
        % Select rho by K-fold cross-validation
        %% -------------------------------------------------

        cv_score = ...
            zeros(length(rho_grid),1);

        for ir = 1:length(rho_grid)

            rho = ...
                rho_grid(ir);

            fold_score = ...
                zeros(K,1);

            for kfold = 1:K

                idx_train = ...
                    training(cv,kfold);

                idx_test = ...
                    test(cv,kfold);

                %% training sufficient statistic

                c_train = ...
                    sum(Zx(idx_train,:),1)';

                N_train = ...
                    sum(idx_train);

                %% validation sufficient statistic

                c_test = ...
                    sum(Zx(idx_test,:),1)';

                N_test = ...
                    sum(idx_test);

                %% fit on the training sample

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
                        opts_fmin);

                %% validation negative log-likelihood

                s = ...
                    Zq*z_hat;

                smax = ...
                    max(s);

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

            cv_score(ir) = ...
                mean(fold_score);
        end

        %% -------------------------------------------------
        % Optimal penalization parameter
        %% -------------------------------------------------

        [cv_min,best_idx] = ...
            min(cv_score);

        rho_opt = ...
            rho_grid(best_idx);

        %% -------------------------------------------------
        % Final fit using the complete district sample
        %% -------------------------------------------------

        c_all = ...
            sum(Zx,1)';

        objective = @(z) ...
            ML_1D_grad( ...
                z, ...
                c_all, ...
                N, ...
                Zq, ...
                xq, ...
                P, ...
                rho_opt);

        z_final = ...
            fminunc( ...
                objective, ...
                zeros(m,1), ...
                opts_fmin);

        %% -------------------------------------------------
        % Estimated clr density
        %% -------------------------------------------------

        s_est = ...
            Zq*z_final;

        %% -------------------------------------------------
        % Estimated probability density
        %% -------------------------------------------------

        smax = ...
            max(s_est);

        f_est = ...
            exp(s_est-smax);

        f_est = ...
            f_est / trapz(xq,f_est);

        %% -------------------------------------------------
        % Store results
        %% -------------------------------------------------

        new_row = ...
            table( ...
                region, ...
                N, ...
                K, ...
                d, ...
                g, ...
                rho_opt, ...
                cv_min, ...
                'VariableNames', { ...
                    'region', ...
                    'N', ...
                    'K', ...
                    'd', ...
                    'g', ...
                    'rho_opt', ...
                    'CV'});

        results = ...
            [results; new_row]; %#ok<AGROW>

        fprintf( ...
            '      rho = %.4g, CV = %.4f\n', ...
            rho_opt,cv_min);
    end
end

%% =========================================================
% SUMMARY OVER DISTRICTS
%% =========================================================

summary = ...
    groupsummary( ...
        results, ...
        'g', ...
        {'mean','median'}, ...
        {'rho_opt','CV'});

disp(summary);

%% =========================================================
% SAVE RESULTS
%% =========================================================

writetable( ...
    results, ...
    sprintf( ...
        'application_1D_Pb_d%d_K%d.csv', ...
        d,K));

writetable( ...
    summary, ...
    sprintf( ...
        'application_1D_Pb_d%d_K%d_summary.csv', ...
        d,K));